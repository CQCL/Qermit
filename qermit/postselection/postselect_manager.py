from collections import Counter
from typing import Dict, List, Tuple

from pytket.backends.backendresult import BackendResult
from pytket.circuit import Bit
from pytket.utils.outcomearray import OutcomeArray


class PostselectMgr:
    """Class for tracking and applying post selection to results.
    Includes other methods to analyse the results after post selection.
    """

    def __init__(
        self,
        compute_cbits: List[Bit],
        postselect_cbits: List[Bit],
    ):
        """Initialisation method.

        :param compute_cbits: Bits in the circuit which are not affected
            by post selection.
        :type compute_cbits: List[Bit]
        :param postselect_cbits: Bits on which the post selection is based.
        :type postselect_cbits: List[Bit]
        :raises Exception: Raised if a bit is in both compute_cbits
            and postselect_cbits.
        """

        intersect = set(compute_cbits).intersection(set(postselect_cbits))
        if intersect:
            raise Exception(
                f"{intersect} are post select and compute qubits. "
                + "They cannot be both."
            )

        self.compute_cbits: List[Bit] = compute_cbits
        self.postselect_cbits: List[Bit] = postselect_cbits

        self.cbits: List[Bit] = compute_cbits + postselect_cbits

    def get_postselected_shot(self, shot: Tuple[int, ...]) -> Tuple[int, ...]:
        "Removes postselection bits from shot."
        return tuple(
            [
                bit
                for bit, reg in zip(shot, self.cbits)
                if reg not in self.postselect_cbits
            ]
        )

    def is_postselect_shot(self, shot: Tuple[int, ...]) -> bool:
        "Determines if shot survives postselection"

        # TODO: It may be nice to generalise this so that other functions
        # besides bit==0 can be used as a means of postselection.
        return all(
            bit == 0
            for bit, reg in zip(shot, self.cbits)
            if reg in self.postselect_cbits
        )

    def dict_to_result(self, result_dict: Dict[Tuple[int, ...], int]) -> BackendResult:
        """Convert dictionary to BackendResult.

        :param result_dict: Dictionary to convert.
        :type result_dict: Dict[Tuple[int, ...], int]
        :return: Corresponding BackendResult.
        :rtype: BackendResult
        """

        # Special case where the dictionary is empty. Presently having
        # an empty counter results in an error.
        if not result_dict:
            return BackendResult()

        return BackendResult(
            counts=Counter(
                {
                    OutcomeArray.from_readouts([key]): val
                    for key, val in result_dict.items()
                }
            ),
            c_bits=self.compute_cbits,
        )

    def postselect_result(self, result: BackendResult) -> BackendResult:
        """Transforms BackendResult to keep only shots which should be
        post selected.

        :param result: Result to be modified.
        :type result: BackendResult
        :return: Postselected shots.
        :rtype: BackendResult
        """

        return self.dict_to_result(
            {
                self.get_postselected_shot(shot): count
                for shot, count in result.get_counts(cbits=self.cbits).items()
                if self.is_postselect_shot(shot)
            }
        )

    def merge_result(self, result: BackendResult) -> BackendResult:
        """Transforms BackendResult so that postselection bits are
        removed, but no shots are removed by postselection.

        :param result: Result to be transformed.
        :type result: BackendResult
        :return: Result with postselection bits removed.
        :rtype: BackendResult
        """

        merge_dict: Dict[Tuple[int, ...], int] = {}
        for shot, count in result.get_counts(cbits=self.cbits).items():
            postselected_shot = self.get_postselected_shot(shot)
            merge_dict[postselected_shot] = merge_dict.get(postselected_shot, 0) + count

        return self.dict_to_result(merge_dict)

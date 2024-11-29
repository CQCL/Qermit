from collections import Counter
from typing import Dict, List, Tuple

from pytket.backends.backendresult import BackendResult
from pytket.circuit import Bit
from pytket.utils.outcomearray import OutcomeArray


class PostselectMgr:
    """Class for tracking and applying post selection to results.

    An example use case might be the following. Here a Bell state is
    prepared. We would like to keep one bit of the results, conditioned
    on the other being 0. That's to say that the postselected
    bits should all be 0.

    .. jupyter-execute::

        from pytket import Circuit, Bit, Qubit
        from pytket.circuit.display import render_circuit_jupyter

        # Two qubits. The result of measuring the first will
        # be used to postselect the result of measuring the second.
        post_q = Qubit(0)
        comp_q = Qubit(1)

        # Construct Bell state preparation circuit.
        circuit = Circuit()
        circuit.add_qubit(post_q)
        circuit.add_qubit(comp_q)

        circuit.H(post_q)
        circuit.CX(post_q, comp_q)

        post_b = Bit(0)
        comp_b = Bit(1)

        circuit.add_bit(post_b)
        circuit.add_bit(comp_b)

        circuit.Measure(comp_q, comp_b)
        circuit.Measure(post_q, post_b)

        render_circuit_jupyter(circuit)

    Running this circuit gives a roughly equal mix of 00 and 11
    computation basis states.

    .. jupyter-execute::

        from qermit.postselection import PostselectMgr
        from pytket.extensions.quantinuum import QuantinuumBackend, QuantinuumAPIOffline

        backend = QuantinuumBackend(
            device_name="H1-1LE",
            api_handler = QuantinuumAPIOffline(),
        )

        compiled_circuit = backend.get_compiled_circuit(circuit)
        result = backend.run_circuit(compiled_circuit, 100)
        result.get_counts()

    """

    def __init__(
        self,
        compute_cbits: List[Bit],
        postselect_cbits: List[Bit],
    ):
        """
        This class is straightforwardly initialised with the computation
        and post selection bits. The computation bits are post selected based
        on the results of the post selection bits.

        .. jupyter-execute::

            postselect_mgr = PostselectMgr(
                compute_cbits=[comp_b],
                postselect_cbits=[post_b]
            )

        :param compute_cbits: Bits in the circuit which are not affected
            by post selection.
        :param postselect_cbits: Bits on which the post selection is based.
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

    def _get_postselected_shot(self, shot: Tuple[int, ...]) -> Tuple[int, ...]:
        "Removes postselection bits from shot."
        return tuple(
            [
                bit
                for bit, reg in zip(shot, self.cbits)
                if reg not in self.postselect_cbits
            ]
        )

    def _is_postselect_shot(self, shot: Tuple[int, ...]) -> bool:
        "Determines if shot survives postselection"

        # TODO: It may be nice to generalise this so that other functions
        # besides bit==0 can be used as a means of postselection.
        return all(
            bit == 0
            for bit, reg in zip(shot, self.cbits)
            if reg in self.postselect_cbits
        )

    def _dict_to_result(self, result_dict: Dict[Tuple[int, ...], int]) -> BackendResult:
        """Convert dictionary to BackendResult.

        :param result_dict: Dictionary to convert. Should be in the form of
            map from shot to count.
        :return: Corresponding BackendResult.
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

        .. jupyter-execute::

            post_result = postselect_mgr.postselect_result(result=result)
            post_result.get_counts()

        :param result: Result to be modified.
        :return: Postselected shots.
        """

        return self._dict_to_result(
            {
                self._get_postselected_shot(shot): count
                for shot, count in result.get_counts(cbits=self.cbits).items()
                if self._is_postselect_shot(shot)
            }
        )

    def merge_result(self, result: BackendResult) -> BackendResult:
        """Transforms BackendResult so that postselection bits are
        removed, but no shots are removed by postselection.

        .. jupyter-execute::

            merge_result = postselect_mgr.merge_result(result=result)
            merge_result.get_counts()

        :param result: Result to be transformed.
        :return: Result with postselection bits removed.
        """

        merge_dict: Dict[Tuple[int, ...], int] = {}
        for shot, count in result.get_counts(cbits=self.cbits).items():
            postselected_shot = self._get_postselected_shot(shot)
            merge_dict[postselected_shot] = merge_dict.get(postselected_shot, 0) + count

        return self._dict_to_result(merge_dict)

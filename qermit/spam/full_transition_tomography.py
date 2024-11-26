# Copyright 2019-2023 Quantinuum
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import itertools
from collections import OrderedDict, namedtuple
from enum import Enum
from functools import lru_cache
from math import ceil, log2
from typing import Counter, Dict, Iterable, List, Optional, Tuple, cast

import numpy as np
from pytket.backends import Backend
from pytket.backends.backendresult import BackendResult
from pytket.circuit import Bit, CircBox, Circuit, Node, Qubit
from pytket.passes import DecomposeBoxes, FlattenRegisters
from pytket.unit_id import UnitID
from pytket.utils.outcomearray import OutcomeArray

FullCorrelatedNoiseCharacterisation = namedtuple(
    "FullCorrelatedNoiseCharacterisation",
    ["CorrelatedNodes", "NodeToIntDict", "CharacterisationMatrices"],
)

StateInfo = namedtuple("StateInfo", ["PreparedStates", "QubitBitMaps"])


# Helper methods for holding basis states
@lru_cache(maxsize=128)
def binary_to_int(bintuple: Tuple[int]) -> int:
    """Convert a binary tuple to corresponding integer, with most significant bit as the
    first element of tuple.

    :param bintuple: Binary tuple
    :return: Integer
    """
    integer = 0
    for index, bitset in enumerate(reversed(bintuple)):
        if bitset:
            integer |= 1 << index
    return integer


@lru_cache(maxsize=128)
def int_to_binary(val: int, dim: int) -> Tuple[int, ...]:
    """Convert an integer to corresponding binary tuple, with most significant bit as
    the first element of tuple.

    :param val: input integer
    :param dim: Bit width
    :return: Binary tuple of width dim
    """
    return tuple(map(int, format(val, "0{}b".format(dim))))


def get_full_transition_tomography_circuits(
    process_circuit: Circuit, backend: Backend, correlations: List[List[Node]]
) -> Tuple[List[Circuit], List[StateInfo]]:
    """Generate calibration circuits according to the specified correlation, backend and given circuit.

    :param circuit: Circuit surmising correlated noise process being characterised
    :param backend: Backend on which the experiments are run.
    :param correlations: A list of lists of correlated Nodes of a `Device`.
        Qubits within the same list are assumed to only have errors correlated
        with each other. Thus to allow errors between all qubits you should
        provide a single list.  The qubits in `correlations` must be nodes in the
        backend's associated `Device`.
    :return: A list of calibration circuits to be run on the machine. The circuits
        should be processed without compilation.
    """

    def to_tuple(correlation_list: List[Node]) -> Tuple[Node, ...]:
        return tuple(correlation_list)

    subsets_matrix_map = OrderedDict.fromkeys(
        sorted(map(to_tuple, correlations), key=len, reverse=True)
    )
    # ordered from largest to smallest via OrderedDict & sorted
    subset_dimensions = [len(subset) for subset in subsets_matrix_map]
    major_state_dimensions = subset_dimensions[0]
    n_circuits = 1 << major_state_dimensions
    all_qubits = [qb for subset in correlations for qb in subset]

    if len(process_circuit.qubits) != len(all_qubits):
        raise ValueError(
            "Process being characterised has {} qubits, correlations only specify {} qubits.".format(
                len(process_circuit.qubits), len(all_qubits)
            )
        )

    # output
    prepared_circuits = []
    state_infos = []

    # set up CircBox of X gate for preparing basis states
    xcirc = Circuit(1).X(0)
    xcirc = backend.get_compiled_circuit(xcirc, optimisation_level=0)
    FlattenRegisters().apply(xcirc)
    xbox = CircBox(xcirc)
    # need to be default register to add as box suitably

    n_qubits_pre_compile = process_circuit.n_qubits
    # This needs to be optimisation level 0 to avoid using simplify initial
    process_circuit = backend.get_compiled_circuit(
        process_circuit, optimisation_level=0
    )

    while process_circuit.n_qubits < n_qubits_pre_compile:
        process_circuit.add_qubit(Qubit("temp_q", process_circuit.n_qubits))

    rename_map_pc = {}
    for index, qb in enumerate(process_circuit.qubits):
        rename_map_pc[qb] = Qubit(index)
    process_circuit.rename_units(cast(Dict[UnitID, UnitID], rename_map_pc))

    pbox = CircBox(process_circuit)

    # set up base circuit for appending xbox to
    base_circuit = Circuit()
    index = 0
    measures = {}
    for qb in all_qubits:
        base_circuit.add_qubit(qb)
        c_bit = Bit(index)
        base_circuit.add_bit(c_bit)
        index += 1
        measures[qb] = c_bit

    # generate state circuits for given correlations
    for major_state_index in range(n_circuits):
        state_circuit = base_circuit.copy()
        # get bit string corresponding to basis state of biggest subset of qubits
        major_state = int_to_binary(major_state_index, major_state_dimensions)
        new_state_dicts = {}
        # parallelise circuits, run uncorrelated subsets characterisation in parallel
        for dim, qubits in zip(subset_dimensions, subsets_matrix_map):
            # add state to prepared states
            new_state_dicts[qubits] = major_state[:dim]
            # find only qubits that are expected to be in 1 state, add xbox to given qubits
            for flipped_qb in itertools.compress(qubits, major_state[:dim]):
                state_circuit.add_circbox(xbox, [cast(UnitID, flipped_qb)])
        # Decompose boxes, add barriers to preserve circuit, add measures
        state_circuit.add_barrier(cast(List[UnitID], all_qubits))

        # add process circuit to measure
        state_circuit.add_circbox(pbox, cast(List[UnitID], state_circuit.qubits))
        DecomposeBoxes().apply(state_circuit)
        state_circuit.add_barrier(cast(List[UnitID], all_qubits))
        for q in measures:
            state_circuit.Measure(q, measures[q])
        # add to returned types
        state_circuit = backend.get_compiled_circuit(state_circuit)
        prepared_circuits.append(state_circuit)
        state_infos.append(StateInfo(new_state_dicts, measures))
    return (prepared_circuits, state_infos)


def calculate_correlation_matrices(
    results_list: List[BackendResult],
    states_info: List[StateInfo],
    correlations: List[List[Node]],
) -> FullCorrelatedNoiseCharacterisation:
    """Calculate the calibration matrices corresponding to some pure noise from the results of running calibration
    circuits.

    :param results_list: List of result via BackendResult. Must be in the same order as the
        corresponding circuits given by prepared_states.
    :param states_info: Each StateInfo object contains the state prepared via a binary
        representation and the qubit_to_bit_map for the corresponding state circuit.
    :param correlations: List of dict corresponding to each prepared basis state

    :return: Characterisation for pure noise given by process circuit
    """

    def to_tuple(correlation_list: List[Node]) -> Tuple[Node, ...]:
        return tuple(correlation_list)

    subsets_matrix_map = OrderedDict.fromkeys(
        sorted(map(to_tuple, correlations), key=len, reverse=True)
    )
    # ordered from largest to smallest via OrderedDict & sorted
    subset_dimensions = [len(subset) for subset in subsets_matrix_map]

    counter = 0
    node_index_dict = dict()
    for qbs, dim in zip(subsets_matrix_map, subset_dimensions):
        # for a subset with n qubits, create a 2^n by 2^n matrix
        subsets_matrix_map[qbs] = np.zeros((1 << dim,) * 2, dtype=float)
        for i in range(len(qbs)):
            qb = qbs[i]
            node_index_dict[qb] = (counter, i)
        counter += 1

    for result, state_info in zip(results_list, states_info):
        state_dict = state_info[0]
        qb_bit_map = state_info[1]
        for qb_sub in subsets_matrix_map:
            # bits of counts to consider
            bits = [qb_bit_map[q] for q in qb_sub]
            counts_dict = result.get_counts(cbits=bits)
            for measured_state, count in counts_dict.items():
                # intended state
                prepared_state_index = binary_to_int(state_dict[qb_sub])
                # produced state
                measured_state_index = binary_to_int(measured_state)
                # update characterisation matrix
                subsets_matrix_map[qb_sub][
                    measured_state_index, prepared_state_index
                ] += count  # type: ignore

    # normalise everything
    normalised_mats = [mat / np.sum(mat, axis=0) for mat in subsets_matrix_map.values()]  # type: ignore
    return FullCorrelatedNoiseCharacterisation(
        correlations, node_index_dict, normalised_mats
    )


# _compute_dot and helper functions #
#
# With thanks to
# https://math.stackexchange.com/a/3423910
# and especially
# https://gist.github.com/ahwillia/f65bc70cb30206d4eadec857b98c4065
# on which this code is based.
def _unfold(tens: np.ndarray, mode: int, dims: List[int]) -> np.ndarray:
    """
    Unfolds tensor into matrix.

    :param tens: Tensor with shape equivalent to dimensions
    :param mode: Specifies axis move to front of matrix in unfolding of tensor
    :param dims: Gives shape of tensor passed

    :return: Matrix with shape (dims[mode], prod(dims[/mode]))
    """
    if mode == 0:
        return tens.reshape(dims[0], -1)
    else:
        return np.moveaxis(tens, mode, 0).reshape(dims[mode], -1)


def _refold(vec: np.ndarray, mode: int, dims: List[int]) -> np.ndarray:
    """
    Refolds vector into tensor.

    :param vec: Tensor with length equivalent to the product of dimensions given in dims
    :param mode: Axis tensor was unfolded along
    :param dims: Shape of tensor

    :return: Tensor folded from vector with shape equivalent to dimensions given in dims
    """
    if mode == 0:
        return vec.reshape(dims)
    else:
        # Reshape and then move dims[mode] back to its
        # appropriate spot (undoing the `unfold` operation).
        tens = vec.reshape([dims[mode]] + [d for m, d in enumerate(dims) if m != mode])
        return np.moveaxis(tens, 0, mode)


def _compute_dot(submatrices: Iterable[np.ndarray], vector: np.ndarray) -> np.ndarray:
    """
    Multiplies the kronecker product of the given submatrices with given vector.

    :param submatrices: Submatrices multiplied
    :param vector: Vector multplied

    :return: Kronecker product of arguments
    """
    dims = [A.shape[0] for A in submatrices]
    vt = vector.reshape(dims)
    for i, A in enumerate(submatrices):
        vt = _refold(A @ _unfold(vt, i, dims), i, dims)
    return vt.ravel()


def _bayesian_iteration(
    submatrices: Iterable[np.ndarray],
    measurements: np.ndarray,
    t: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """
    Transforms T corresponds to a Bayesian iteration, used to modfiy measurements.

    :param submatrices: submatrices to be inverted and applied to measurements.
    :param measurements: Probability distribution over some set of states to be amended.
    :param t: Some transform to act on measurements.
    :param epsilon: A stabilization parameter  to define an affine transformation for applicatoin
    to submatrices, eliminating zero probabilities.

    :return: Transformed distribution vector.
    """
    # Transform t according to the Bayesian iteration
    # The parameter epsilon is a stabilization parameter which defines an affine
    # transformation to apply to the submatrices to eliminate zero probabilities. This
    # transformation preserves the property that all columns sum to 1
    if epsilon == 0:
        # avoid copying if we don't need to
        As = submatrices
    else:
        As = [
            epsilon / submatrix.shape[0] + (1 - epsilon) * submatrix
            for submatrix in submatrices
        ]
    z = _compute_dot(As, t)
    if np.isclose(z, 0).any():
        raise ZeroDivisionError
    return cast(
        np.ndarray, t * _compute_dot([A.transpose() for A in As], measurements / z)
    )


def _bayesian_iterative_correct(
    submatrices: Iterable[np.ndarray],
    measurements: np.ndarray,
    tol: float = 1e-5,
    max_it: Optional[int] = None,
) -> np.ndarray:
    """
    Finds new states to represent application of inversion of submatrices on measurements.
    Converges when update states within tol range of previously tested states.

    :param submatrices: Matrices comprising the pure noise characterisation.
    :param input_vector: Vector corresponding to some counts distribution.
    :param tol: tolerance of closeness of found results
    :param max_it: Maximum number of inversions attempted to correct results.
    """
    # based on method found in https://arxiv.org/abs/1910.00129

    vector_size = measurements.size
    # uniform initial
    true_states = np.full(vector_size, 1 / vector_size)
    prev_true = true_states.copy()
    converged = False
    count = 0
    epsilon: float = 0  # stabilization parameter, adjusted dynamically
    while not converged:
        if max_it:
            if count >= max_it:
                break
            count += 1
        try:
            true_states = _bayesian_iteration(
                submatrices, measurements, true_states, epsilon
            )
            converged = np.allclose(true_states, prev_true, atol=tol)
            prev_true = true_states.copy()
        except ZeroDivisionError:
            # Shift the stabilization parameter up a bit (always < 0.5).
            epsilon = 0.99 * epsilon + 0.01 * 0.5

    return true_states


class CorrectionMethod(Enum):
    def Invert(
        submatrices: Iterable[np.ndarray], input_vector: np.ndarray
    ) -> np.ndarray:
        """
        Multiplies the kronecker product of given submatrices on input vector
        and then adjusts output to make them genuine probabilities. Submatrices
        represent pure noise characterisation, vector corresponds to counts
        distribution from circuit and device.

        :param submatrices: Matrices comprising the pure noise characterisation.
        :param input_vector: Vector corresponding to some counts distribution.
        """
        try:
            subinverts = [np.linalg.inv(submatrix) for submatrix in submatrices]
        except np.linalg.LinAlgError:
            raise ValueError(
                "Unable to invert calibration matrix: please re-run "
                "calibration experiments or use an alternative correction method."
            )
        # assumes that order of rows in flattened subinverts equals order of bits in input vector
        v = _compute_dot(subinverts, input_vector)
        # The entries of v will always sum to 1, but they may not all be in the range [0,1].
        # In order to make them genuine probabilities (and thus generate meaningful counts),
        # we adjust them by setting all negative values to 0 and scaling the remainder.
        v[v < 0] = 0
        v /= sum(v)
        return v

    def Bayesian(
        submatrices: Iterable[np.ndarray], input_vector: np.ndarray
    ) -> np.ndarray:
        """
        Computes the product of the invert of submatrices on the given input vector via a an iterative
        Bayesian correction method.

        :param submatrices: Matrices comprising the pure noise characterisation.
        :param input_vector: Vector corresponding to some counts distribution.
        :param tol: tolerance of closeness of found results
        :param max_it: Maximum number of inversions attempted to correct results.
        """
        return _bayesian_iterative_correct(
            submatrices, input_vector, tol=1e-5, max_it=500
        )


def reduce_matrix(indices_to_remove: List[int], matrix: np.ndarray) -> np.ndarray:
    """
    Removes indices from indices_to_remove from binary associated to indexing of matrix,
    producing a new transition matrix.
    To do so, it assigns all transition probabilities as the given state in the remaining
    indices binary, with the removed binary in state 0. This is an assumption on the noise made
    because it is likely that unmeasured qubits will be in that state.

    :param indices_to_remove: Binary index of state matrix is mapping to be removed.
    :param matrix: Transition matrix where indices correspond to some binary state, to have some
    dimension removed.

    :return: Transition matrix with removed entries.
    """

    new_n_qubits = int(log2(matrix.shape[0])) - len(indices_to_remove)
    if new_n_qubits == 0:
        return np.array([])
    bin_map = dict()
    mat_dim = 1 << new_n_qubits
    for index in range(mat_dim):
        # get current binary
        bina = list(int_to_binary(index, new_n_qubits))
        # add 0's to fetch old binary to set values from
        for i in sorted(indices_to_remove):
            bina.insert(i, 0)
        # get index of values
        bin_map[index] = binary_to_int(tuple(bina))

    new_mat = np.zeros((mat_dim,) * 2, dtype=float)
    for i in range(len(new_mat)):
        old_row_index = bin_map[i]
        for j in range(len(new_mat)):
            old_col_index = bin_map[j]
            new_mat[i, j] = matrix[old_row_index, old_col_index]
    return new_mat


def reduce_matrices(
    entries_to_remove: List[Tuple[int, int]], matrices: List[np.ndarray]
) -> List[np.ndarray]:
    """
    Removes some dimensions from some matrices.
    :param entries_to_remove: Via indexing, says which dimensions to remove from which indices.
    :param matrices: All matrices to have dimensions removed.

    :return: Matrices with some dimensions removed.
    """
    organise: Dict[int, List[int]] = {k: [] for k in range(len(matrices))}
    for unused in entries_to_remove:
        # unused[0] is index in matrices
        # unused[1] is qubit index in matrix
        organise[unused[0]].append(unused[1])
    output_matrices = [reduce_matrix(organise[m], matrices[m]) for m in organise]
    normalised_mats = [
        mat / np.sum(mat, axis=0) for mat in [x for x in output_matrices if x.size > 0]
    ]
    return normalised_mats


def get_single_matrix(
    entry_to_keep: Tuple[int, int], matrices: List[np.ndarray]
) -> np.ndarray:
    """
    Returns a correction matrix just for the index given.
    :param entry_to_keep: Which matrix and indexing to return a correction matrix for.
    :param matrices: All matrices to find returned matrix from.

    :return: Matrix for correcting given entry.
    """
    mat = matrices[entry_to_keep[0]]
    all_indices = list(range(int(log2(mat.shape[0]))))
    all_indices.remove(entry_to_keep[1])
    return reduce_matrix(all_indices, mat)


def correct_transition_noise(
    result: BackendResult,
    bit_qb_info: Tuple[Dict[Qubit, Bit], Dict[Bit, Qubit]],
    noise_characterisation: FullCorrelatedNoiseCharacterisation,
    corr_method: CorrectionMethod,
) -> BackendResult:
    """
    Modifies count distribution for result, such that the inversion of the pure noise map represented by
    matrices in noise_characterisation is applied to it

    :param result: BackendResult object to be negated by pure noise object.
    :param bit_qb_info: Used to permute corresponding BackendResult object so counts order matches noise characterisation.
    :param noise_characterisation: Object holding all required information for some full noise characterisation of correlated subsets.
    """

    final_measures_qb_map = bit_qb_info[0]
    mid_circuit_measures_bq_map = bit_qb_info[1]
    # get counts from with order of bits that matches order of qubits in subsets
    # if qubit in subset has no bit, skip it
    char_bits_order = []
    unused_final_qbs = []
    for subset in noise_characterisation.CorrelatedNodes:
        for q in subset:
            if q in final_measures_qb_map:
                char_bits_order.append(final_measures_qb_map[q])
            else:
                unused_final_qbs.append(q)

    mid_measure_qbs = []
    for bit in mid_circuit_measures_bq_map:
        mid_measure_qbs.append(mid_circuit_measures_bq_map[bit])
        char_bits_order.append(bit)

    # get counts object for returning later
    counts = result.get_counts(cbits=char_bits_order)
    in_vec = np.zeros(1 << len(char_bits_order), dtype=float)
    # turn from counts to probability distribution
    for state, count in counts.items():
        in_vec[binary_to_int(state)] = count
    Ncounts = np.sum(in_vec)
    in_vec_norm = in_vec / Ncounts
    # remove correlated qubits with no bits in results from matrices for correcting
    # this step just for measures in "final" stage
    reduced_matrices = reduce_matrices(
        [noise_characterisation.NodeToIntDict[q] for q in unused_final_qbs],
        noise_characterisation.CharacterisationMatrices,
    )
    # now added additional matrices to list for correct midcircuit measurements
    for q in mid_measure_qbs:
        reduced_matrices.append(
            get_single_matrix(
                noise_characterisation.NodeToIntDict[q],
                noise_characterisation.CharacterisationMatrices,
            )
        )

    # with counts and characterisation matrices orders matching, correct distribution
    # enum type not callable, ignore
    outvec = corr_method(reduced_matrices, in_vec_norm)  # type: ignore
    outvec *= Ncounts
    # counter object with binary from distribution
    corrected_counts = {
        int_to_binary(index, len(char_bits_order)): Bcount
        for index, Bcount in enumerate(outvec)
    }
    counter = Counter(
        {
            OutcomeArray.from_readouts([key]): ceil(val)
            for key, val in corrected_counts.items()
        }
    )
    # produce and return BackendResult object
    return BackendResult(counts=counter, c_bits=char_bits_order)

# Copyright 2019-2021 Cambridge Quantum Computing
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


from typing import Counter, Tuple, List, Dict, Set, cast, Sequence
from pytket.circuit import Circuit, Qubit, Bit, CircBox  # type: ignore
from pytket.passes import DecomposeBoxes, FlattenRegisters  # type: ignore
from qermit import MitTask, CircuitShots
import itertools
from collections import namedtuple
from pytket.backends import Backend
from pytket.backends.backendresult import BackendResult
from pytket.utils.outcomearray import OutcomeArray
import numpy as np
from math import ceil
import networkx as nx  # type: ignore


PartialCorrelatedNoiseCharacterisation = namedtuple(
    "PartialCorrelatedNoiseCharacterisation",
    ["Distance", "CorrelatedEdges", "CorrelationToMatrix"],
)


def gen_state_polarisation_dicts(correlations: nx.Graph) -> List[Dict]:
    """Generating all four basis states for capturing two-qubit correlations requires circuits
    producing the following states:
    A B
    0 0
    0 1
    1 0
    1 1
    Correlations between two-qubits can be assumed captured for a given process if tomography circuits
    are run preparing these basis states. A and B also function as two patterns for initialisating qubits;
    if 4 circuits are produced for Qubit 1 with pattern A, and the same circuits are extended to
    initialise some Qubits 2 and 3 in pattern B, then when run for some process, the given circuits
    will capture all correlations betweeen Qubits 1 and 2 and Qubits 1 and 3, and only half (00 & 11)
    for Qubits 2 and 3. If correlations between 2 and 3 need to be captured, a new set of circuits must be run
    with 2 in pattern A and 3 in pattern B (or vice versa). We can summarise this information with the following
    rows as:
    1 2 3
    A B B
    2 3
    A B
    Translated to basis state preparation, this would require 8 circuits, 4 for each row (though n.b. that with
    later optimisation this can be reduced to 6, as the second row only needs to capture 01 and 10 correlations for
    Qubits 2 and 3).
    For any Graph, where nodes are qubits and edges capture desired correlations between qubits, this algorithm
    finds Qubit & pattern combinations covering all correlations with a number of rows equal to the
    number of nodes in the largest fully connected subgraph - 1 (n.b. each row then corresponds to 4 basis states
    meaning 4*(number of nodes in the largest fully connected subgraph - 1) circuits are required to characterise
    correlated noise)(also n.b. this is with the assumption that strong correlations between adjacent qubits
    does not imply weak correlations between said adjacent qubits)

    :param correlations: Graph object wherein adjacent nodes have two qubit correlations
    :type correlations: nx.Graph

    :return: A list of dictionaries, each dictionary from node to basis state preparation pattern
    :rtype: List[Dict[UnitID, bool]]
    """
    # dicts for holding polarisation pattern
    state_polarisation_dicts: List[Dict] = []
    used_nodes: Set[Qubit] = set()
    for node in sorted(set(correlations.nodes)):
        # also prevents case where neighbour==node causing issue (some hardware graphs are defined in this way...)
        used_nodes.add(node)
        for neighbour in sorted(set(correlations.neighbors(node))):
            # corelations are bidirectional as is graph
            # check prevents correlations being added twice
            if neighbour not in used_nodes:
                value_set = False
                # note state_polarisation_dicts is a list, longest polarisation will always be first entry
                for states in state_polarisation_dicts:
                    # break to not add correlations to multiple states! (though this will happen naturally in some cases and is fine)
                    if node not in states:
                        # node not in states => neighbour not in states
                        # as it implies neighbour has not been node yet
                        states[node] = True
                        states[neighbour] = False
                        value_set = True
                        break
                    if node in states and neighbour not in states:
                        # set to opposite polarisation
                        states[neighbour] = not states[node]
                        value_set = True
                        break
                    if node in states and neighbour in states:
                        if states[node] != states[neighbour]:
                            # correlation already captured
                            value_set = True
                            break
                if not value_set:
                    # correlation can't be captured with any existing state, prep a new one
                    state_polarisation_dicts.append(
                        dict([(node, True), (neighbour, False)])
                    )
    # Sort as number of circuits run total can be minimised further later with help of reverse ordering
    state_polarisation_dicts.sort(key=len)
    return state_polarisation_dicts


def gen_partial_tomography_circuits(
    state_polarisations: List[Dict], x_prep_box: CircBox
) -> Tuple[List[Circuit], Dict]:
    """Basis states for correlations between two qubits can be prepared with the following patterns:
    A B
    0 0
    0 1
    1 0
    1 1
    state_polarisations gives a List[Dict], for which each dictionary is from UnitID to pattern.
    gen_partial_tomography_circuits converts these patterns to circuits, with basis state prep given by x_prep_bo (i.e.
    gate primitive for X for desired hardware). Returned Dict has tuple((UnitID, UnitID),(int, int)) as keys and int
    as values, instructing later characterisation procedures to which BackendResult are required to capture some
    correlation between two qubits.

    :param state_polarisations: Guides basis state construction, as given by gen_state_polarisation_dicts
    :type state_polarisations: List[Dict]
    :param x_prep_box: Single qubit CircBox giving Pyktet Circuit for preparing |1> state for a single qubit
    :type x_prep_box: CircBox

    :return: Characterisation circuits, and dict for finding results for correlation basis and Qubits
    :rtype: Tuple(Tuple(UnitID, UnitID), Tuple(int, int))
    """
    all_circuits: List[Circuit] = []
    correlations_dict = dict()
    for states in state_polarisations:
        # for two qubits there are 4 possible basis states
        basis_state_circuits = [Circuit(), Circuit(), Circuit(), Circuit()]
        # for later organsiation all correlations a given 'state polarisation; captures
        true_keys: Set[Qubit] = set()
        false_keys: Set[Qubit] = set()
        # track measurements as they are added
        measure_dict: Dict[Qubit, Bit] = dict()

        for node in states:
            # add qubit to each basis state circuit
            for c in basis_state_circuits:
                c.add_qubit(node)
            # if true, implies qubit needs to be in 0 state for first two basis states, and 1 state for second two
            if states[node]:
                true_keys.add(node)
                basis_state_circuits[2].add_circbox(x_prep_box, [node])
                basis_state_circuits[3].add_circbox(x_prep_box, [node])
            # if false, implies qubit needs to be in 0 state for first and third basis states, and 1 state for second and last
            else:
                false_keys.add(node)
                basis_state_circuits[1].add_circbox(x_prep_box, [node])
                basis_state_circuits[3].add_circbox(x_prep_box, [node])
            # add entry to measurement dict, measure qubits later
            measure_dict[node] = Bit(len(measure_dict))

        # make circuit proper for running -> decompose circuit box, add physical barrier to prevent further optimisation
        for c in basis_state_circuits:
            DecomposeBoxes().apply(c)
            c.add_barrier(c.qubits)
        # add measures to circuit
        for k in measure_dict:
            v = measure_dict[k]
            for c in basis_state_circuits:
                c.add_bit(v)
                c.Measure(k, v)

        # fill correlations dict
        base_len = len(all_circuits)
        for t in sorted(true_keys):
            for f in sorted(false_keys):
                # keys a pair of circuit qubits pertaining to some correlation
                # values are Index in results, state measured and classical bits required to retrieve correlation result
                correlations_dict[(t, f)] = [
                    (base_len, (0, 0), (measure_dict[t], measure_dict[f])),
                    (base_len + 1, (0, 1), (measure_dict[t], measure_dict[f])),
                    (base_len + 2, (1, 0), (measure_dict[t], measure_dict[f])),
                    (base_len + 3, (1, 1), (measure_dict[t], measure_dict[f])),
                ]
        # add circuits to all circuits
        for c in basis_state_circuits:
            all_circuits.append(c)

    return (all_circuits, correlations_dict)


def partial_spam_setup_task_gen(backend: Backend, correlation_distance: int) -> MitTask:
    """
    Sets up required information for characterising and correcting SPAM noise.
    Includes check for whether SPAM Characterisation needs to be run.

    :param backend: Default Backend characterisation and experiment are executed on.
    :type backend: Backend
    :param correlations_distance: Distance over Backend Connectivity graph over which correlations in Qubit SPAM Noise is expected.
    :type correlations_distance: int

    :return: A MitTask object that completes said task.
    :rtype: MitTask
    """

    def task(
        obj, circ_shots: List[CircuitShots]
    ) -> Tuple[List[CircuitShots], List[Dict[Qubit, Bit]], bool]:
        characterisation = True

        if correlation_distance < 1:
            raise ValueError(
                "Correlations distance must be 1 or greater for characterisation of partial spam correlations."
            )
        if backend.backend_info is None:
            raise ValueError("Backend has no characterisation attribute.")
        else:
            if "CorrelatedSpamCorrection" in backend.backend_info.misc:
                # check correlations distance
                if (
                    backend.backend_info.misc["CorrelatedSpamCorrection"].Distance
                    is correlation_distance
                ):
                    characterisation = False
        qubit_bit_maps = [c[0].qubit_to_bit_map for c in circ_shots]
        return (circ_shots, qubit_bit_maps, characterisation)

    return MitTask(_label="SPAMSetup", _n_in_wires=1, _n_out_wires=3, _method=task)


def partial_correlated_spam_circuits_task_gen(
    backend: Backend, calibration_shots: int, correlation_distance: int
) -> MitTask:
    """
    For Given Backend, returns circuits characterisating SPAM noise for up to correlation_distance distance on the Backend connectivity graph.

    :param backend: Default Backend characterisation and experiment are executed on.
    :type backend: Backend
    :param calibration_shots: Number of shots required for each characterisation circuit
    :type calibration_shots: int
    :param correlations_distance: Distance over Backend Connectivity graph over which correlations in Qubit SPAM Noise is expected.
    :type correlations_distance: int

    :return: A MitTask object that produces Characterisation circuits.
    :rtype: MitTask
    """

    def task(obj, characterise: bool) -> Tuple[List[CircuitShots], Dict]:
        """
        Pure Function that transforms experimental circuit information to characterisation circuits and dict
        for storing correlations for SPAM noise.

        :param characterise: True if characterisation required
        :type characterise: bool

        :return: A list of Characterisation circuits and dict from correlation to index in Characterisation Circuits list
        and classical bits required derive characterisation for correlation.
        :rtype: Tuple[List[CircuitShots], Dict]

        """
        if characterise:
            # get basic graph, representing nearest neighbour correlations
            correlations = nx.Graph()

            if backend.backend_info is None:
                raise ValueError("Backend has no backend_info attribute.")
            correlations.add_edges_from(backend.backend_info.architecture.coupling)
            # to allow non-nearest neighbour correlations, add edges to graph
            # to reflect larger distance correlations
            for _ in itertools.repeat(None, correlation_distance - 1):
                new_correlations = []
                for node in sorted(set(correlations.nodes)):
                    # Forced to spell this way by itertools...
                    neighbors = sorted(set(correlations.neighbors(node)))
                    # NoiseModel Backends and some Hardware Backends Will have
                    # Couplings specified between a Node to itself
                    if node in neighbors:
                        neighbors.remove(node)
                    for pair in itertools.combinations(neighbors, r=2):
                        new_correlations.append(pair)
                correlations.add_edges_from(new_correlations)

            # gen basis state prep polarisations required to cover all correlations
            sp_dicts = gen_state_polarisation_dicts(correlations)

            # produce basis state preparation circuit
            xcirc = Circuit(1).X(0)
            xcirc = backend.get_compiled_circuit(xcirc)
            FlattenRegisters().apply(xcirc)
            x_prep_box = CircBox(xcirc)

            # get tomography circuits
            circs_dict = gen_partial_tomography_circuits(sp_dicts, x_prep_box)
            # translate circuits to CircShots type required by wire
            circ_shots = [
                CircuitShots(Circuit=c, Shots=calibration_shots) for c in circs_dict[0]
            ]
            return (circ_shots, circs_dict[1])
        else:
            return ([], dict())

    return MitTask(
        _label="SPAMCalibrationCircuits", _n_in_wires=1, _n_out_wires=2, _method=task
    )


def order_counts(counts: Counter[Tuple[int, ...]]) -> List[int]:
    """
    Helper method for organising two-qubit correlations.
    :param counts: Counter object as returned by BackendResult, giving two qubit counts for desired correlation
    :type counts: Counter[Tuple[int, ...]]

    :return: A four element list, giving counts for the (0,0), (0,1), (1,0) and (1,1) states in order.
    :rtype: List[int]
    """
    ordered_counts = [0, 0, 0, 0]
    if (0, 0) in counts:
        ordered_counts[0] = counts[(0, 0)]
    if (0, 1) in counts:
        ordered_counts[1] = counts[(0, 1)]
    if (1, 0) in counts:
        ordered_counts[2] = counts[(1, 0)]
    if (1, 1) in counts:
        ordered_counts[3] = counts[(1, 1)]
    return ordered_counts


def characterise_correlated_spam_task_gen(
    backend: Backend, correlations_distance: int, calibration_shots: int
) -> MitTask:
    """
    Produces SPAM characterisation matrices for each correlation expected.

    :param backend: Backend which characterisation to be stored in
    :type backend: Backend
    :param correlations_distance: Correlation Distance which charactersation was produced for - only required for setting in backend.
    :type correlations_distance: int

    :return: A MitTask object that assigns a PartialCorrelatedNoiseCharacterisation to backend.
    :rtype: MitTask

    """

    def task(obj, results: List[BackendResult], correlations_dict: Dict) -> Tuple[bool]:
        """
        Pure function that transforms BackendResult objects to a PartialCorrelatedNoiseCharacterisation that is stored
        in backend, only returning a bool for confirming this task has completed.

        :param results: Results retrieved from defined backend.
        :type results: List[BackendResult]
        :param correlations_dict: For a given graph edge, gives the index of results for which results for a given correlation ((0,0) or (0,1) etc) can be found from given bits.
        :type correlations_dict: dict
        """

        # if condition not met, nothing is run as characterisation is not required
        # guarantees characterisation isn't written over
        if len(results) > 0 and len(correlations_dict) > 0:
            correlation_to_mat = dict()
            correlated_edges = []
            for nodes in correlations_dict:
                value = correlations_dict[nodes]
                cal_mat = np.empty((4, 4))
                correlated_edges.append(nodes)
                # 00->0; 01->1; 10->2; 11->3; measured->row; prepared->column
                cal_mat[:, 0] = order_counts(
                    results[value[0][0]].get_counts(list(value[0][2]))
                )
                cal_mat[:, 1] = order_counts(
                    results[value[1][0]].get_counts(list(value[1][2]))
                )
                cal_mat[:, 2] = order_counts(
                    results[value[2][0]].get_counts(list(value[2][2]))
                )
                cal_mat[:, 3] = order_counts(
                    results[value[3][0]].get_counts(list(value[3][2]))
                )
                # normalised matrix
                cal_mat = cal_mat / calibration_shots
                v = np.linalg.inv(cal_mat)
                v[v < 0] = 0
                v /= sum(v)
                correlation_to_mat[nodes] = v

            if backend.backend_info is None:
                raise ValueError("Backend has no backend_info attribute.")
            backend.backend_info.add_misc(
                "CorrelatedSpamCorrection",
                PartialCorrelatedNoiseCharacterisation(
                    correlations_distance, correlated_edges, correlation_to_mat
                ),
            )
        return (True,)

    return MitTask(
        _label="SPAMCharacterisation", _n_in_wires=2, _n_out_wires=1, _method=task
    )


def collate_corrected_counts(
    width: int,
    c_bits: Dict[Bit, int],
    correlation_counts: List[Tuple[Tuple[Bit, Bit], List[int], int]],
    total_counts: int,
) -> BackendResult:
    """
    Converts a list of counts for different pairs of indices into a single BackendResult object. Width must be
    greater than largest index (assume 0 indexing).

    :param width: Number of bits in output result
    :type width: int
    :param corrected_counts: Each entry gives counts for a different set of four basis states for the given Bits.
    :type corrected_counts: List[Tuple(Tuple(Bit, Bit), List[int], int)]

    :return: A BackendResult object capturing all these counts
    :rtype: BackendResult
    """
    collated_counts = dict.fromkeys(itertools.product([0, 1], repeat=width), float(1))
    # For each correlation, retrieve bits corrected counts are for
    for correlation in correlation_counts:
        indices = [c_bits[correlation[0][0]], c_bits[correlation[0][1]]]
        # division acts as normalisation, retrieve counts
        counts = [x / correlation[2] for x in correlation[1]]
        # update dictionary containing all counts
        # due to symmetry of binary strings, each string will have the same number of counts added for each correlations

        # need to find allow states from given, and proportion
        # i.e. if n shots taken, find proportion of given state and return thins info
        # possible states?
        for key in collated_counts:
            # if else statements for assigning corrected counts to binary representing given count
            if key[indices[0]] == 0:
                if key[indices[1]] == 0:
                    collated_counts[key] = counts[0] * collated_counts[key]
                else:
                    collated_counts[key] = counts[1] * collated_counts[key]
            elif (
                key[indices[1]] == 0
            ):  # not prior if => key[indices[0]] == 1, so just check second bit though maybe safer to check...
                collated_counts[key] = counts[2] * collated_counts[key]
            else:  # every other option exhausted, so must be this
                collated_counts[key] = counts[3] * collated_counts[key]

    # convert collated counts to a BackendResult object
    normalisation_factor = sum(collated_counts.values())
    # use first shots value in list i.e. correlation_counts[0][2]
    for k in collated_counts:
        collated_counts[k] = (collated_counts[k] / normalisation_factor) * total_counts
    counter = Counter(
        {
            OutcomeArray.from_readouts([key]): ceil(val)
            for key, val in collated_counts.items()
        }
    )
    return BackendResult(counts=counter, c_bits=cast(Sequence[Bit], c_bits))


def correct_partial_correlated_spam_task_gen(
    backend: Backend,
) -> MitTask:
    """
    Produces and returns a MitTask object to apply SPAM correction to BackendResult objects.
    :param backend: Backend from which Spam Correction characterisation is retrieved.
    :type backend: Backend
    """

    def task(
        obj,
        results: List[BackendResult],
        qubit_readouts: List[Dict],
        characterised: bool,
    ) -> Tuple[List[BackendResult]]:

        if backend.backend_info is None:
            raise ValueError("Backend has no backend_info attribute.")
        if "CorrelatedSpamCorrection" in backend.backend_info.misc:
            cnc = backend.backend_info.misc["CorrelatedSpamCorrection"]
        else:
            raise ValueError(
                "'CorrelatedSpamCorrection' not characterised for Backend."
            )
        output_results = []
        for result, qubit_readout in zip(results, qubit_readouts):
            desired_qubits = qubit_readout.keys()
            single_correlation_corrections = []
            total_counts = len(result.get_shots())
            for edge in cnc.CorrelatedEdges:
                # only get correlated qubit edge if actually measured
                # if not checked, wouldn't be able to retrieve bits for qubits not present
                if edge[0] in desired_qubits and edge[1] in desired_qubits:
                    # edge in terms of Qubit, find bit as provided by experiment Circuit
                    bits = (qubit_readout[edge[0]], qubit_readout[edge[1]])
                    # get correction matrix from Backend characterisation (already inverted)
                    mat = cnc.CorrelationToMatrix[edge]
                    # order counts converts key -> value counts with a list where 0th index gives 00 result, 1st 01, 2nd 10 and 3rd 11.
                    edge_counts = order_counts(result.get_counts(list(bits)))
                    # correct this list for SPAM with the inverted correction matrix
                    corrected_counts = mat.dot(edge_counts)
                    # add corrected counts for given correlation in terms of bits results represent, along with total shots
                    single_correlation_corrections.append(
                        (bits, corrected_counts, sum(edge_counts))
                    )
            # convert list of 2 qubit counts for different correlated bits into a single BackendResult

            corrected_result = collate_corrected_counts(
                len(result.c_bits),
                result.c_bits,
                single_correlation_corrections,
                total_counts,
            )

            output_results.append(corrected_result)

        return (output_results,)

    return MitTask(_label="SPAMCorrection", _n_in_wires=3, _n_out_wires=1, _method=task)

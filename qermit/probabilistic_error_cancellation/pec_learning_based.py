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

from qermit import (
    MitEx,
    MitRes,
    AnsatzCircuit,
    MitTask,
    ObservableTracker,
    ObservableExperiment,
    TaskGraph,
)
from qermit.zero_noise_extrapolation.zne import (
    gen_duplication_task,
    gen_initial_compilation_task,
    gen_qubit_relabel_task,
)
from qermit.probabilistic_error_cancellation.cliff_circuit_gen import (
    random_clifford_circ,
)

from sympy.core.expr import Expr

from pytket.passes import RebaseTket, DecomposeBoxes  # type: ignore
from pytket.utils import QubitPauliOperator, get_pauli_expectation_value
from pytket.backends import Backend
from pytket.transform import Transform  # type: ignore
from pytket.circuit import Op, CircBox, OpType, Circuit, Node  # type: ignore
from pytket.placement import place_with_map  # type: ignore

from pytket.pauli import QubitPauliString  # type: ignore
from pytket.predicates import CliffordCircuitPredicate  # type: ignore

import re
from typing import List, Tuple, Dict, cast, Union, Any
from copy import copy
import numpy as np

QuasiProbabilities = List[float]


def str_to_pauli_op(pauli_str: str) -> Op:
    """
    Returns Pauli operator corresponding to given string

    :param pauli_str: one of 'X','Y','Z','I'
    :type pauli_str: str
    :return: Pauli operator
    :rtype: Op
    """
    assert pauli_str in ["X", "Z", "Y", "I"]
    switcher = {
        "X": Op.create(OpType.X),
        "Z": Op.create(OpType.Z),
        "Y": Op.create(OpType.Y),
        "I": Op.create(OpType.noop),
    }
    return switcher.get(pauli_str)


def random_commuting_clifford(
    circ: Circuit,
    qps: QubitPauliString,
    simulator_backend: Backend,
    max_count: int = 1000,
    n_shots: int = 1000,
) -> Circuit:
    """Replace all Computing gates with random Clifford gates. The expectation
    of the given Pauli string on the final Clifford circuit is non-zero.

    :param circ: Initial circuit. Should include gates labelled as Computing using opgroups.
    :type circ: Circuit
    :param qps: Pauli string which should have non-zero expectation on the outputted circuit.
    :type qps: QubitPauliString
    :param max_count: Maximum number of attempts at finding a Clifford circuit with non-zero expectation value.
    :type max_count: int
    :param simulator_backend: Backend for deriving Pauli Expectation values
    :type simulator_backend: Backend
    :raises ValueError: Raised if the circuit does not include any gates labelled as Computing.
    :raises RuntimeError: Raised if no replacement of Computing gates with Clifford gates could be found such
        that the expectation of the inputted Pauli string is non-zero.
    :raises RuntimeError: Raised if the resulting circuit is not Clifford. This could be because
        not all Computing gates in the original circuit were labelled as such.
    :return: Clifford circuit, build by replacing Computing gates with random Clifford gates.
    :rtype: Circuit
    """

    # Build list of all opgroup names corresponding to Computing gates.
    comp_opgroup_list = [
        i["opgroup"] for i in circ.to_dict()["commands"] if "Computing" in i["opgroup"]
    ]

    count = 0

    # Check if the circuit contains any Computing gates.
    if len(comp_opgroup_list) == 0:
        raise ValueError(
            "This circuit contains no computing gates (i.e. single qubit gates). Training is not possible."
        )

    # Repeats until a replacement of all Computing gates with Clifford gates which results
    # in a circuit with non-zero expectation value is found, or the maximum number
    # of iterations is exceeded.
    expect_val = complex(0)
    while round(abs(expect_val)) == 0:
        rand_cliff_circ = circ.copy()

        # Retrieve a list of random Clifford gates, one for each of
        # the Computing gates in the original circuit. Note this is in the form of a
        # CircBox so that the opgroup labels persist after substitution
        # (they would not do so if circuits were used instead of CircBox)
        rand_cliff_list = [CircBox(random_clifford_circ(1)) for _ in comp_opgroup_list]
        # Replace Computing gates with Clifford gates.
        for opgroup, rand_cliff in zip(comp_opgroup_list, rand_cliff_list):
            rand_cliff_circ.substitute_named(rand_cliff, opgroup)

        DecomposeBoxes().apply(rand_cliff_circ)

        # Check if the expectation of the given Pauli string is non-zero on the Clifford
        # circuit. Leave while loop if so.

        n_q_map = dict()
        cc_qns = rand_cliff_circ.qubits
        for i in range(len(cc_qns)):
            n_q_map[cc_qns[i]] = Node("q", i)

        new_qps_qbs = []
        qps_paulis = []
        qps_dict = qps.map
        for x in qps_dict:
            new_qps_qbs.append(n_q_map[x])
            qps_paulis.append(qps_dict[x])

        new_qps = QubitPauliString(new_qps_qbs, qps_paulis)

        place_with_map(rand_cliff_circ, n_q_map)

        # Check if state is supported, otherwise use shots, otherwise raise error
        if simulator_backend.supports_state:
            expect_val = get_pauli_expectation_value(
                rand_cliff_circ, new_qps, simulator_backend
            )
        elif simulator_backend.supports_shots or simulator_backend.supports_counts:
            expect_val = get_pauli_expectation_value(
                rand_cliff_circ, new_qps, simulator_backend, n_shots=n_shots
            )
        else:
            raise RuntimeError(
                "The simulator backend does not support state, shots or counts."
            )
        # TODO: Better management of the case that there are no circuits with expectation value not equal to 0.

        # Check if the number of attempts at finding a circuit with non-zero expectation exceeds the maximum.
        count += 1
        if count == max_count:
            raise RuntimeError(
                "Could not find circuit with non-zero expectation. It's possible there are none."
            )

    # Verify that the resulting circuit is a Clifford circuit.
    if not CliffordCircuitPredicate().verify(rand_cliff_circ):
        raise RuntimeError(
            "The resulting circuit is not a Clifford circuit. This could be because not all Computing gates were labelled as such."
        )

    return rand_cliff_circ


def substitute_pauli(circ: Circuit, frame_name: str, pauli_pair: List[str]) -> Circuit:
    """
    Replace 2 qubit Pauli gate pair which surrounds Frame gate with a 2 qubit pauli gate and its inverse.

    :param circ: Initial circuit. This should include a gate in the opgroup frame_name.
    :type circ: Circuit
    :param frame_name: The opgroup of the Frame gate that the pauli gates will act either side of
    :type frame_name: str
    :param pauli_pair: Two strings describing the 2 qubit Pauli gate
    :type pauli_pair: List[str]
    :return: A circuit, with the Pauli gates inserted either side of the given frame gate.
    :rtype: Circuit
    """

    match_return = re.match(r"Frame (.*)", frame_name)
    if match_return is None:
        raise ValueError(
            "The name of this frame gate does not match the form 'Frame i', where i is an integer indexing the gate"
        )
    frame_number = int(match_return.group(1))

    substituted_circ = circ.copy()

    # post and pre describes which side of the frame gate is being considered
    for pos in ("post", "pre"):
        # Add Pauli gate to the first and second qubit of the Frame gate
        for qubit in (0, 1):
            substituted_circ.substitute_named(
                pauli_pair[qubit], "%s Pauli %i %i" % (pos, frame_number, qubit)
            )
    return substituted_circ


def substitute_pauli_but_one(
    circ: Circuit, to_replace_opgroup: str, pauli_pair: List[str]
) -> Circuit:
    """Sets all Pauli gates to the identity, apart from those around the
    inputted Frame gate, described by its opgroup.

    :param circ: Initial Circuit
    :type circ: Circuit
    :param to_replace_opgroup: The opgroup of the frame gate who's corresponding
        Pauli gates should be replaced with the inputted gate.
    :type to_replace_opgroup: str
    :param pauli_pair: Two strings describing the Pauli gate to be substituted in
    :type pauli_pair: List[str]
    :raises RuntimeError: Raised if the inputted circuit does not have a gate in the opgroup inputted.
    :return: The final circuit with the Pauli gates substituted in.
    :rtype: Circuit
    """

    # Gather list of all of the opgroup of Frame gates in the circuit.
    frame_opgroup_list = [
        i["opgroup"] for i in circ.to_dict()["commands"] if "Frame" in i["opgroup"]
    ]
    # Raise error if there is no circuits in the opgroup given as input.
    if not (to_replace_opgroup in frame_opgroup_list):
        raise RuntimeError(
            "No Frame Gate with given name %s in circuit" % to_replace_opgroup
        )

    # Remove gate to substitute with inputted Pauli gate from list. Now the list
    # consists of those gates that should be substituted with identity.
    frame_opgroup_list.remove(to_replace_opgroup)

    # Substitute inputted Pauli gate.
    substituted_circ = substitute_pauli(circ, to_replace_opgroup, pauli_pair)

    # Substitute all remaining Pauli gates with the identity.
    for opgroup in frame_opgroup_list:
        substituted_circ = substitute_pauli(
            substituted_circ, opgroup, [str_to_pauli_op("I"), str_to_pauli_op("I")]
        )
    return substituted_circ


def PECRebase(circ: Circuit) -> Circuit:
    """Rebase circuit so that all multi-qubit gates are Clifford gates.
    All consecutive single qubit gates should be compressed to one single qubit gate.

    :param circ: Initial circuit in any gate set.
    :type circ: Circuit
    :return: Rebased circuit
    :rtype: Circuit
    """
    rebased_circ = circ.copy()
    RebaseTket().apply(rebased_circ)
    Transform.ReduceSingles().apply(rebased_circ)
    return rebased_circ


def gen_rebase_to_frames_and_computing() -> MitTask:
    """Generates task which rebases circuits into the Frame and Computing gate set,
    as defined in https://arxiv.org/abs/2005.07601.
    In particular, all two qubit gates are Clifford gates (called Frame gates),
    and all single qubit gates (called Computing gates) are merged where possible.

    :return: MitTask object which rebases circuits into the Frame and Computing gate set.
    :rtype: MitTask
    """

    def task(
        obj, wire: List[ObservableExperiment]
    ) -> Tuple[List[ObservableExperiment]]:
        """
        Transform all circuits into the Frame and Computing gates, as defined in https://arxiv.org/abs/2005.07601.
        In particular, all two qubit gates should be Clifford gates (called Frame gates),
        and all single qubit gates (called Computing gates) should be merged where possible

        :param wire: Circuits
        :type wire: List[ObservableExperiment]
        :return: Rebased Circuits
        :rtype: Tuple[List[ObservableExperiment]]
        """

        framed_circ_list = []

        for obs_exp in wire:
            # rebase circuit
            framed_circ = PECRebase(obs_exp.AnsatzCircuit.Circuit)
            framed_circ_list.append(
                ObservableExperiment(
                    AnsatzCircuit=AnsatzCircuit(
                        Circuit=framed_circ,
                        Shots=obs_exp.AnsatzCircuit.Shots,
                        SymbolsDict=obs_exp.AnsatzCircuit.SymbolsDict,
                    ),
                    ObservableTracker=obs_exp.ObservableTracker,
                )
            )

        return (framed_circ_list,)

    return MitTask(
        _label="RebasePEC",
        _n_out_wires=1,
        _n_in_wires=1,
        _method=task,
    )


# TODO: Add task before this to collate results using noisy_list_structure
def gen_run_with_quasi_prob() -> MitTask:
    """Generates task which converts list of quasi probabilities and noisy circuit results
    into error mitigated results.

    :return: MitTask performing mitigation given noisy results and quasi-probabilities.
    :rtype: MitTask
    """

    def task(
        obj,
        prob_list: List[List[QuasiProbabilities]],
        noisy_circ_results: List[QubitPauliOperator],
        noisy_list_structure: List[Dict[str, int]],
    ) -> Tuple[List[QubitPauliOperator]]:
        """Converts list of quasi probabilities and noisy circuit results into error mitigated results.

        :param prob_list: List of quasi probabilities for error-mitigation. Note that the outer
            most list corresponds to circuits, the second level list corresponds to qubit pauli strings,
            and the inner most list contains the quasi probabilities.
        :type prob_list: List[List[QuasiProbabilities]], OuterList all experiments, second level list
            for MeasurementCircuits, third level list is for QuasiProbabilities
        :param noisy_circ_results: List of MitEx results from noisy backend.
        :type noisy_circ_results: Tuple[List[QubitPauliOperator]]
        :param noisy_list_structure: Information on the structure of the noisy results.
            This takes the form {'experiment':int, 'error':int}, where 'experiment' indexes the
            initial circuit and 'error' indexes the Pauli error applied.
        :type noisy_list_structure: List[Dict[str, int]]
        :return: Error mitigated results.
        :rtype: Tuple[List[QubitPauliOperator]]
        """

        circ_results_list = []
        # Creates new list of noisy results to match the form of the list
        # of quasi probabilities. Each inner list now consists of results
        # corresponding to one ideal circuit.
        first = 0
        for experiment in prob_list:
            # The number of noisy circuits corresponding to one ideal.
            last = len(experiment[0])
            # Append portion of the list of noisy results that relate to one circuit.
            circ_results_list.append(noisy_circ_results[first : first + last])
            first += last

        # Create new list with mitigated results
        em_expect_list = []
        for circuit_results, circuit_prob in zip(circ_results_list, prob_list):
            em_expect = {}
            for i, qps in enumerate(circuit_results[0]._dict):
                em_expect[qps] = 0
                # Sum noisy results, adjusted by quasi probabilities.
                for j, prob in enumerate(circuit_prob[i]):
                    em_expect[qps] += prob * circuit_results[j][qps]
            em_expect_qpo = QubitPauliOperator(
                cast(Dict[Any, Union[int, float, complex, Expr]], em_expect)
            )
            em_expect_list.append(em_expect_qpo)

        return (em_expect_list,)

    return MitTask(
        _label="MitigateWithQuasiProbs",
        _n_out_wires=1,
        _n_in_wires=3,
        _method=task,
    )


def collate_results_task_gen() -> MitTask:
    """Generates task which collates results from running circuit, and circuits with frame
    gates wrapped in Pauli gates. The results are collated so as to facilitate
    learning the quasiprobabilities required for correction. The data itself is
    not changed by this task.

    :return: MitTask object collating results.
    :rtype: MitTask
    """

    def task(
        obj,
        noisy_results: List[QubitPauliOperator],
        noisy_list_structure: List[Dict[str, int]],
        ideal_results: List[QubitPauliOperator],
        ideal_list_structure: List[Dict[str, int]],
    ) -> Tuple[List[List[List[List[Tuple[QubitPauliOperator, QubitPauliOperator]]]]]]:
        """Collates results from implementing circuit and circuits with frame gates wrapped in Pauli gates.
        We call these respectively ideal and noisy. The results are collated so as to facilitate learning
        the quasiprobabilities required for correction. The data itself is not changed by this task.

        :param noisy_results: List of expectations values of noisy Clifford circuits.
        :type noisy_results: List[QubitPauliOperator]
        :param noisy_list_structure: Information on the structure of the noisy results.
            This takes the form {'experiment':int, 'error':int}, where 'experiment' indexes the
            initial circuit and 'error' indexes the Pauli error applied.
        :type noisy_list_structure: List[Dict[str, int]]
        :param ideal_results: List of expectations values of ideal Clifford circuits
        :type ideal_results: List[QubitPauliOperator]
        :param ideal_list_structure: Information on the structure of the ideal results.
            This takes the form {'experiment':int, 'qps':int, 'training circuit':int},
            where 'experiment' indexes the initial circuit, 'qps' indexes the QubitPauliString,
            and 'training circuit' indexes the random Clifford derived from the initial circuit.
        :type ideal_list_structure: List[Dict[str,int]]
        :return: Collated results. The inner most Tuple corresponds to expectation results for a
            fixed ObservableExperiment, QubitPauliString, clifford circuit, and Pauli noise.
            Each list level fixes consecutively an ObservableExperiment, QubitPauliString, and Clifford circuit.
        :rtype: Tuple[List[List[List[List[Tuple[QubitPauliOperator, QubitPauliOperator]]]]]]
        """

        if not len(ideal_list_structure) == len(ideal_results):
            raise RuntimeError(
                "The length of the list structure and list of ideal results do not match."
            )
        if not len(noisy_list_structure) == len(noisy_results):
            raise RuntimeError(
                "The length of the list structure and list of noisy results do not match."
            )

        fixed_clifford_nn_experiment_operators = []

        # Gather all experiment indexes amongst the ideal results.
        all_obs_exp_index = set()
        for obs_exp in ideal_list_structure:
            all_obs_exp_index.add(obs_exp["experiment"])

        for obs_exp_index in all_obs_exp_index:

            # For each experiment index, create a list of the list structure information
            # of results corresponding to that experiment index.
            fixed_obs_exp_details = [
                (i, ideal_exp_dict)
                for i, ideal_exp_dict in enumerate(ideal_list_structure)
                if ideal_exp_dict["experiment"] == obs_exp_index
            ]

            # Gather all QubitPauliString indexes amongst the ideal results
            # with a fixed experiment index.
            all_qps_index = set()
            for _, obs_exp_dict in fixed_obs_exp_details:
                all_qps_index.add(obs_exp_dict["qps"])

            fixed_obs_exp_results = []

            for qps_index in all_qps_index:

                # For each QubitPauliString index, create a list of the list structure information
                # of results corresponding to that QubitPauliString index.
                fixed_qps_details = [
                    ideal_exp_dict
                    for ideal_exp_dict in fixed_obs_exp_details
                    if ideal_exp_dict[1]["qps"] == qps_index
                ]

                fixed_qps_results = []

                for cliff_details in fixed_qps_details:

                    # For each Clifford circuit corresponding to a fixed experiment and
                    # QuasiPauliString, gather the noisy results list structure information.
                    fixed_cliff_details = [
                        (i, noisy_exp_dict)
                        for i, noisy_exp_dict in enumerate(noisy_list_structure)
                        if noisy_exp_dict["experiment"] == cliff_details[0]
                    ]

                    fixed_cliff_results = []

                    for noisy_details in fixed_cliff_details:

                        # For each noisy circuit corresponding to a fixed experiment,
                        # QubitPauliString and Clifford circuit, gather the noisy ideal result pair.
                        fixed_cliff_results.append(
                            (
                                noisy_results[noisy_details[0]],
                                ideal_results[cliff_details[0]],
                            )
                        )

                    fixed_qps_results.append(fixed_cliff_results)

                fixed_obs_exp_results.append(fixed_qps_results)

            fixed_clifford_nn_experiment_operators.append(fixed_obs_exp_results)

        return (fixed_clifford_nn_experiment_operators,)

    return MitTask(
        _label="CollateResults",
        _n_out_wires=1,
        _n_in_wires=4,
        _method=task,
    )


def learn_quasi_probs_task_gen(num_cliff_circ: int) -> MitTask:
    """
    Generates task which characterises quasi-probabilities. This takes ideal simulation results
    (from running Clifford circuit) and noisy results, and uses
    them to deduce quasi-probabilities for later correction of real experiment results.

    :param num_cliff_circ: The number of Clifford circuits generated for each inputted circuit.
    :type num_cliff_circ: int

    :return: MitTask object for producing quasi probabilities.
    :rtype: MitTask
    """

    def task(
        obj,
        results: List[List[List[Tuple[QubitPauliOperator]]]],
    ) -> Tuple[List[List[QuasiProbabilities]]]:
        """This implementation of learning base probabilistic error cancellation is
         based on the significant error approach of https://arxiv.org/abs/2005.07601

        :param results: Collated results. The inner most Tuple corresponds to
            expectation results for a fixed ObservableExperiment, QubitPauliString,
            clifford circuit, and Pauli noise. Each list level fixes consecutively
            an ObservableExperiment, QubitPauliString, and clifford circuit.
        :type results: List[ List[List[Tuple[QubitPauliOperator]]] ]
        :return: List of quasi probabilities. The outer list corresponds to circuits,
            the second level list corresponds to Pauli strings, and the inner most
            list corresponds to quasi probabilities.
        :rtype: Tuple[List[List[QuasiProbabilities]]]
        """

        prob_list = []
        # qps_results is List[List[Tuple[QubitPauliOperator, QubitPauliOperator]]]
        # each tuple is a noisy, noiseless pair of results for perturbed fixed Clifford circuit
        # each inner list is results for fixed clifford circuit
        # each outerlist is for a single Qubit Pauli String in experiment
        for qps_results in results:

            qps_quasi_prob_list = []
            # qps is List[List[Tuple[QubitPauliOperator, QubitPauliOperator]]]
            # containing all results for all fixed Clifford circuits
            # required in characterisation experiments
            for qps in qps_results:
                num_pauli_gates = len(qps[0])
                a = np.zeros((num_pauli_gates, num_pauli_gates))
                b = np.zeros(num_pauli_gates)
                # cliff is List[Tuple[QubitPauliOperator, QubitPauliOperator]]
                # noisy and noiseless results for each pertubred Clifford circuit run
                # perturbation being from base fixed Clifford circuit
                for cliff in qps:
                    # Iterate over the possible Pauli noises, calculating elements of the
                    # matrices needed for methods of least square
                    # P is a pair of noisy and noiseless qubit pauli operators
                    for i, P in enumerate(cliff):
                        noisy_i_qpo = P[0]
                        ideal_i_qpo = P[1]
                        # TODO: can this one liner be shrunk?
                        noisy_P_expectation = noisy_i_qpo._dict[
                            list(noisy_i_qpo._dict.keys())[0]
                        ]
                        ideal_P_expectation = ideal_i_qpo._dict[
                            list(ideal_i_qpo._dict.keys())[0]
                        ]
                        b[i] += (
                            (1 / num_cliff_circ)
                            * noisy_P_expectation
                            * ideal_P_expectation
                        )

                        for j, Q in enumerate(cliff):
                            # TODO: Check why this doesn't need ideal?
                            noisy_j_qpo = Q[0]
                            noisy_Q_expectation = noisy_j_qpo._dict[
                                list(noisy_j_qpo._dict.keys())[0]
                            ]
                            a[i, j] += (
                                (1 / num_cliff_circ)
                                * noisy_P_expectation
                                * noisy_Q_expectation
                            )

                # Invert matrix a in order to find quasi probabilities of each noise
                # each qubit pauli string has an associated quasi probability for correction
                qps_quasi_prob_list.append(list(np.linalg.lstsq(a, b, rcond=None)[0]))
            # qps_quasi_prob_list will have a quasi probability for each qubit pauli string
            # in the experiments QubitPauliOperator
            # prob_list holds this information for all QubitPauliOperators for all experiments
            prob_list.append(qps_quasi_prob_list)

        return (prob_list,)

    return MitTask(
        _label="LearnQuasiProbs",
        _n_out_wires=1,
        _n_in_wires=1,
        _method=task,
    )


def gen_get_clifford_training_set(
    simulator_backend: Backend, num_rand_cliff: int
) -> MitTask:
    """
    Generates task which creates characterisation Clifford circuits. These circuits are
    constructed from an initial circuit by replacing all Computing gates with random Clifford gates.

    :param simulator_backend: Ideal simulator backend on which Clifford circuits are to be run.
    :type simulator_backend: Backend
    :param num_rand_cliff: Number of random Clifford circuits for each fixed ObservableExperiment.
    :type num_rand_cliff: int

    :return: MitTask object for producing random Clifford circuits.
    :rtype: MitTask
    """

    def task(
        obj, wire: List[ObservableExperiment]
    ) -> Tuple[List[ObservableExperiment], List[Dict[str, int]]]:
        """Create a list of Clifford circuits built from the input circuits by
        randomly replacing each Computing gate with a Clifford gate.

        :param wire: Initial circuits
        :type wire: List[Tuple[AnsatzCircuit,ObservableTracker]]
        :return: Clifford circuits
        :rtype: Tuple[List[ObservableExperiment]]
        """

        training_circ_list = []

        list_structure_info = []

        for experiment_num, experiment in enumerate(wire):
            ansatz_circuit = experiment.AnsatzCircuit
            qpo = experiment.ObservableTracker.qubit_pauli_operator

            for qps_num, qps in enumerate(qpo._dict):
                # Generate a list of circuits such that each Computing gate
                # is replaced by a random Clifford gate.
                training_circs = [
                    random_commuting_clifford(
                        ansatz_circuit.Circuit, qps, simulator_backend
                    )
                    for i in range(num_rand_cliff)
                ]

                for training_circuit_num, training_circuit in enumerate(training_circs):
                    cliff_ansatz_circuit = AnsatzCircuit(
                        Circuit=training_circuit,
                        Shots=ansatz_circuit.Shots,
                        SymbolsDict=ansatz_circuit.SymbolsDict,
                    )
                    cliff_tracker = ObservableTracker(QubitPauliOperator({qps: 1}))
                    training_circ_list.append(
                        ObservableExperiment(
                            AnsatzCircuit=cliff_ansatz_circuit,
                            ObservableTracker=cliff_tracker,
                        )
                    )
                    list_structure_info.append(
                        {
                            "experiment": experiment_num,
                            "qps": qps_num,
                            "training_circuit": training_circuit_num,
                        }
                    )

        return (
            training_circ_list,
            list_structure_info,
        )

    return MitTask(
        _label="CliffordTrainingSet",
        _n_out_wires=2,
        _n_in_wires=1,
        _method=task,
    )


def label_gates(circ: Circuit) -> Circuit:
    """Label all of the gates in the circuit as with "Frame" or "Computing".
    The label includes an index to describe the ordering of the gates

    :param circ: Circuit which should be in the TK1, CX basis
    :type circ: Circuit
    :raises RuntimeError: Raised if the circuit is not in the required basis.
    :return: Identical circuit, but with gates assigned opgroups.
    :rtype: Circuit
    """

    # Recover list of commands describing initial circuit.
    circ_dict = circ.to_dict()
    command_list = circ_dict["commands"]

    # Add labels, in the form of opgroups, to each of the commands in the list
    labelled_command_list = []
    comp_count = 0
    frame_count = 0
    for command in command_list:
        labelled_command = command.copy()
        if labelled_command["op"]["type"] in ("TK1"):
            labelled_command["opgroup"] = "Computing %i" % comp_count
            comp_count += 1
        elif labelled_command["op"]["type"] in ("CX"):
            labelled_command["opgroup"] = "Frame %i" % frame_count
            frame_count += 1
        else:
            raise RuntimeError(
                'This gate is not one of either "TK1" or "CX". Please ensure you have run PECRebase before using this function.'
            )
        labelled_command_list.append(labelled_command)

    # Construct new circuit from the list of labelled commands.
    labelled_circ_dict = circ_dict.copy()
    labelled_circ_dict["commands"] = labelled_command_list
    labelled_circ = Circuit().from_dict(labelled_circ_dict)

    return labelled_circ


def gen_label_gates() -> MitTask:
    """Generates task which labels all gates as either Computing or Frame.
    Frame gates are 2-qubit Clifford gates and Computing gates are single qubit gates.
    Circuits should be rebased to Frame and Computing before this task.

    :return: MitTask performing labelling of gates.
    :rtype: MitTask
    """

    def task(
        obj, wire: List[ObservableExperiment]
    ) -> Tuple[List[ObservableExperiment]]:
        """Returns identical circuits but with each gate labelled as Computing or Frame.

        :param wire: Circuits
        :type wire: List[ObservableExperiment]
        :return: Identical circuits with Computing and Frame gates labelled as such.
        :rtype: Tuple[List[ObservableExperiment]]
        """

        labelled_circ_list = []

        for experiment in wire:
            labelled_circ = label_gates(experiment.AnsatzCircuit.Circuit)
            labelled_circ_list.append(
                ObservableExperiment(
                    AnsatzCircuit=AnsatzCircuit(
                        Circuit=labelled_circ,
                        Shots=experiment.AnsatzCircuit.Shots,
                        SymbolsDict=experiment.AnsatzCircuit.SymbolsDict,
                    ),
                    ObservableTracker=experiment.ObservableTracker,
                )
            )

        return (labelled_circ_list,)

    return MitTask(
        _label="LabelGates",
        _n_out_wires=1,
        _n_in_wires=1,
        _method=task,
    )


def wrap_frame_gates(circ: Circuit) -> Circuit:
    """Inserts Pauli gates either side of every Frame gate.
    Initially these are identity gates, so acting as placeholders for later operations.

    :param circ: Initial circuit. Each gate should be labelled as either
        a Frame gate, or a Computing gates.
    :type circ: Circuit
    :raises RuntimeError: Raised if the gates in the circuit are not labelled as
        either a Frame or Computing gate.
    :return: Circuit identical to the original, but with identity gates,
        labelled as Pauli gates, added on either side of every Frame gate.
    :rtype: Circuit
    """

    # Recover list of commands from circuit
    circ_dict = circ.to_dict()
    circ_command_list = circ_dict["commands"]

    framed_circ_command_list = []

    for command in circ_command_list:

        # Add command to new list if not a Frame gate.
        if "Computing" in command["opgroup"]:
            framed_circ_command_list.append(command.copy())

        elif "Frame" in command["opgroup"]:

            match_return = re.match(r"Frame (.*)", command["opgroup"])
            if match_return is None:
                raise ValueError(
                    "The name of this frame gate does not match the form 'Frame i', where i is an integer indexing the gate"
                )
            frame_number = int(match_return.group(1))

            # initially each frame gate has identities added on either side
            pauli = {"op": {"type": "noop"}, "args": "temp", "opgroup": "temp"}

            # Add an identity gate to each qubit on which the Frame gate acts,
            # before the Frame gate itself acts.
            for qubit_i, arg in enumerate(command["args"]):
                labelled_EM_command = pauli.copy()
                labelled_EM_command["args"] = [arg]
                labelled_EM_command["opgroup"] = "pre Pauli %i %i" % (
                    frame_number,
                    qubit_i,
                )
                framed_circ_command_list.append(labelled_EM_command)

            framed_circ_command_list.append(command.copy())

            # After the frame gates is acted, act and identity on each of
            # the qubits acted on by the Frame gate.
            for qubit_i, arg in enumerate(command["args"]):
                labelled_EM_command = pauli.copy()
                labelled_EM_command["args"] = [arg]
                labelled_EM_command["opgroup"] = "post Pauli %i %i" % (
                    frame_number,
                    qubit_i,
                )
                framed_circ_command_list.append(labelled_EM_command)

        else:
            raise RuntimeError(
                'Unrecognised opgroup. Must be either "Computing" or "Frame". This function is called by a gen_wrap_frame_gates task.'
            )

    # Build new circuit from new list of commands.
    framed_circ_dict = circ_dict.copy()
    framed_circ_dict["commands"] = framed_circ_command_list
    framed_circ = Circuit().from_dict(framed_circ_dict)

    return framed_circ


def gen_wrap_frame_gates() -> MitTask:
    """Generates task which wraps Frame gates in Pauli
    gates, initially set to the identity. Pauli gates are labelled as such.

    :return: MitTask which performs wrapping.
    :rtype: MitTask
    """

    def task(
        obj, wire: List[ObservableExperiment]
    ) -> Tuple[List[ObservableExperiment]]:
        """Returns identical circuit but with Frame gates wrapped in Pauli
        gates, initially set to the identity.

        :param wire: Initial circuits
        :type wire: List[ObservableExperiment]
        :return: Circuits with each Frame gate wrapped in identity gates, labelled as Pauli gates.
        :rtype: Tuple[List[ObservableExperiment]]
        """

        framed_circ_list = []
        for experiment in wire:
            framed_circ = wrap_frame_gates(experiment.AnsatzCircuit.Circuit)
            framed_circ_list.append(
                ObservableExperiment(
                    AnsatzCircuit=AnsatzCircuit(
                        Circuit=framed_circ,
                        Shots=experiment.AnsatzCircuit.Shots,
                        SymbolsDict=experiment.AnsatzCircuit.SymbolsDict,
                    ),
                    ObservableTracker=experiment.ObservableTracker,
                )
            )

        return (framed_circ_list,)

    return MitTask(
        _label="WrapFrameGates",
        _n_out_wires=1,
        _n_in_wires=1,
        _method=task,
    )


def list_pauli_gates(circ: Circuit) -> List[Dict]:
    """Produces a list of all possible Pauli errors, assuming an
    error occurs on at mores Frame gate.

    :param circ: Circuit with every gate labelled as a Frame of Computing gate
    :type circ: Circuit
    :raises RuntimeError: Raised if there are no Frame gates in the circuit
    :return: A list of dictionaries describing the errors. Note that as we are assuming at most
        one error in the circuit, it is enough to specify the 2-qubit Pauli error and the gate
        (specified by its opgroup) on which it acts.
    :rtype: List[Dict]
    """

    # Create list of all Frame gate opgroups
    frame_opgroup_list = [
        i["opgroup"] for i in circ.to_dict()["commands"] if "Frame" in i["opgroup"]
    ]

    prob_list = []

    if len(frame_opgroup_list) <= 0:
        raise RuntimeError("There are no Gates of the Frame optype in this circuit.")

    # Add identity to list of possible errors (i.e. no error)
    prob_list.append({"op": ["I", "I"], "opgroup": frame_opgroup_list[0]})

    # To the list of possible errors, add every possible 2-qubit pauli gate
    # that could act before and after a Frame gate. Note that as we are assuming
    # the error acts on at most one Frame gate, it is enough to specify the error
    # and the Frame gate on which it acts.
    for opgroup in frame_opgroup_list:

        for q1_pauli in ["X", "Y", "Z", "I"]:
            for q2_pauli in ["X", "Y", "Z", "I"]:

                if (q1_pauli == "I") and (q2_pauli == "I"):
                    continue

                opgroup_prob_dict = {}
                opgroup_prob_dict["op"] = [q1_pauli, q2_pauli]
                opgroup_prob_dict["opgroup"] = opgroup

                prob_list.append(opgroup_prob_dict)

    return prob_list


def gen_get_noisy_circuits(backend: Backend, **kwargs) -> MitTask:
    """Generates task which create list of circuts, build from original by adding an error
    to one of the Frame gates. An error here is recreated by replacing a pair of Pauli gates,
    wrapped around one Frame gate, by a Pauli gate. Note that there will be a new circuit for each
    possible Pauli error.

    :param backend: Backend on which circuits will be run. Required for compilation.
    :type backend: Backend
    :return: MitTask produsing noisy gates.
    :rtype: MitTask
    """

    def task(
        obj, wire: List[ObservableExperiment]
    ) -> Tuple[List[ObservableExperiment], List[Dict[str, int]]]:
        """Create list of circuts, build from original by adding an error
         to one of the Frame gates. Note that there will be a new circuit for each
         possible Pauli error.

        :param wire: Initial Circuits
        :type wire: List[ObservableExperiment]
        :return: Circuits with pauli gates added around Frame gates to simulate noise.
        :rtype: Tuple[List[ObservableExperiment]]
        """

        list_structure = []

        noisy_circuit_list = []
        # For each circuit, create an equivalent circuit but on which one of the
        # possible errors occur.
        for experiment_num, experiment in enumerate(wire):

            pauli_errors = list_pauli_gates(experiment.AnsatzCircuit.Circuit)

            for error_num, error in enumerate(pauli_errors):
                pauli_circ = substitute_pauli_but_one(
                    experiment.AnsatzCircuit.Circuit,
                    error["opgroup"],
                    [str_to_pauli_op(error["op"][0]), str_to_pauli_op(error["op"][1])],
                )

                pauli_circ = backend.get_compiled_circuit(
                    pauli_circ, optimisation_level=0
                )

                new_ansatz_circuit = AnsatzCircuit(
                    Circuit=pauli_circ,
                    Shots=copy(experiment.AnsatzCircuit.Shots),
                    SymbolsDict=copy(experiment.AnsatzCircuit.SymbolsDict),
                )
                new_tracker = ObservableTracker(
                    experiment.ObservableTracker.qubit_pauli_operator
                )
                noisy_circuit_list.append(
                    ObservableExperiment(
                        AnsatzCircuit=new_ansatz_circuit, ObservableTracker=new_tracker
                    )
                )
                list_structure.append(
                    {"experiment": experiment_num, "error": error_num}
                )

        return (
            noisy_circuit_list,
            list_structure,
        )

    return MitTask(
        _label=kwargs.get("_label", "GetNoisyCircuits"),
        _n_out_wires=2,
        _n_in_wires=1,
        _method=task,
    )


def gen_PEC_learning_based_MitEx(
    device_backend: Backend, simulator_backend: Backend, **kwargs
) -> MitEx:
    """Generates MitEx object for mitigating errors using learning based Probabilistic
    Error Cancellation (PEC), as introduced in https://arxiv.org/abs/2005.07601.

    :param device_backend: Noisy backend on which circuits are to be run.
    :type device_backend: Backend
    :param simulator_backend: Ideal state vector simulator used for simulating Clifford Circuits.
    :type simulator_backend: Backend

    :key simulator_mitex: MitEx object ideal state simulations are run on, default simulator_backend.
    :key device_mitex: MitEx object observable experiments are run on, default device_backend.
    :key seed: Seed for np.random, default None.
    :key optimisation_level: Optimisation level for initial compilation, default 0.
    :key num_cliff: The number of random Clifford circuits generated for each primary circuit, default 10.

    :raises RuntimeError: Raised if the backend gate set does not include CX or CZ gates.
    :return: MitEx object implementing error-mitigation via learning based PEC.
    :rtype: MitEx
    """

    # Disallow backends that do not have 2 qubit clifford gates
    if not (
        (OpType.CX in device_backend.backend_info.gate_set)  # type: ignore
        or (OpType.CZ in device_backend.backend_info.gate_set)  # type:ignore
    ):
        raise RuntimeError("The backend gate set must include CX or CZ gates")

    _seed = kwargs.get("seed", None)
    np.random.seed(seed=_seed)

    _optimisation_level = kwargs.get("optimisation_level", 0)
    # TODO: Change to a number of clifford circuits which varies with the size of the circuit
    num_cliff_circ = kwargs.get("num_cliff", 10)

    sim_mitex = copy(
        kwargs.get(
            "simulator_mitex", MitEx(simulator_backend, _label="IdealCliffordMitEx")
        )
    )

    device_mitres = MitRes(device_backend)
    device_mitex = copy(
        kwargs.get(
            "device_mitex",
            MitEx(device_backend, _label="NoisyMitex", mitres=device_mitres),
        )
    )

    _experiment_taskgraph = TaskGraph().from_TaskGraph(device_mitex)

    _experiment_taskgraph.add_wire()

    get_noisy_clifford_circuits = gen_get_noisy_circuits(
        device_backend, _label="GetNoisyCliffordCircuits"
    )
    _experiment_taskgraph.prepend(get_noisy_clifford_circuits)

    _experiment_taskgraph.parallel(sim_mitex)

    _experiment_taskgraph.prepend(gen_duplication_task(2, _label="DuplicateClifford"))

    _experiment_taskgraph.add_wire()

    get_clifford_training_set = gen_get_clifford_training_set(
        simulator_backend, num_cliff_circ
    )
    _experiment_taskgraph.prepend(get_clifford_training_set)

    collate_results = collate_results_task_gen()
    _experiment_taskgraph.append(collate_results)

    learn_dist = learn_quasi_probs_task_gen(num_cliff_circ)
    _experiment_taskgraph.append(learn_dist)

    _circuit_experiment_taskgraph = TaskGraph().from_TaskGraph(device_mitex)
    _circuit_experiment_taskgraph.add_wire()
    get_noisy_circuits = gen_get_noisy_circuits(
        device_backend, _label="GetNoisyCircuits"
    )
    _circuit_experiment_taskgraph.prepend(get_noisy_circuits)

    _experiment_taskgraph.parallel(_circuit_experiment_taskgraph)

    run_with_probs = gen_run_with_quasi_prob()
    _experiment_taskgraph.append(run_with_probs)

    _experiment_taskgraph.prepend(gen_duplication_task(2, _label="DuplicateCircuits"))

    initial_compilation = gen_initial_compilation_task(
        device_backend, _optimisation_level
    )
    label_gates = gen_label_gates()
    wrap_frame_gates = gen_wrap_frame_gates()
    compile_to_frames_and_computing = gen_rebase_to_frames_and_computing()

    _experiment_taskgraph.prepend(wrap_frame_gates)
    _experiment_taskgraph.prepend(label_gates)
    _experiment_taskgraph.prepend(compile_to_frames_and_computing)

    _experiment_taskgraph.add_wire()
    _experiment_taskgraph.prepend(initial_compilation)
    _experiment_taskgraph.append(gen_qubit_relabel_task())

    return MitEx(device_backend).from_TaskGraph(_experiment_taskgraph)

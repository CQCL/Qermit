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


from copy import copy
from typing import Dict, List, Tuple, Union, cast

from numpy import mean
from pytket import Circuit
from pytket.backends import Backend
from pytket.pauli import Pauli, QubitPauliString
from pytket.tailoring import apply_clifford_basis_change
from pytket.transform import Transform
from pytket.utils import QubitPauliOperator
from sympy.core.expr import Expr  # type: ignore

from qermit import (
    AnsatzCircuit,
    MeasurementCircuit,
    MitEx,
    MitTask,
    ObservableExperiment,
    ObservableTracker,
    SymbolsDict,
    TaskGraph,
)
from qermit.taskgraph.mitex import gen_compiled_MitRes, get_basic_measurement_circuit


def get_clifford_mcs(input_circuit: Circuit) -> List[MeasurementCircuit]:
    """
    For given Circuit, rebases and substitutes all non-Clifford angles with symbols.
    Then, makes MeasurementCircuit objects wherein each Circuit is the same, but each SymbolsDict
    object varies with some different set of Clifford angles.

    :param input_circuit: Circuit to make all-non Clifford gates paramterised.

    :return: New MeasurementCircuits with Clifford parameters
    """
    copy_circ = input_circuit.copy()
    symbols = copy_circ.free_symbols()
    symbols_dict = dict()
    for s in symbols:
        symbols_dict[s] = cast(Union[None, float], float(0))
    sd = SymbolsDict.symbols_from_dict(symbols_dict)
    return [MeasurementCircuit(copy_circ, sd)]


# TODO: make prepare -1 eigenstate also
def preparation_circuit_for_partition(
    clifford_circuit: Circuit, partition: List[QubitPauliString]
) -> Circuit:
    """
    For each Pauli string in partition, finds Pauli string produced by applying a Clifford basis change from given Clifford circuit.
    Returns a state preparation circuit for preparing a +1 eigenstate of all basis changed Pauli strings.
    """

    eigenstate_circuit = Circuit(0)
    for q in clifford_circuit.qubits:
        eigenstate_circuit.add_qubit(q)
    # to make +1 eigenstate of all transformed strings
    # transform string, then append gates for preparing
    # +1 eigenstate to circuit
    # use RemoveRedundancies after to minimise gates required in cosntruction
    Transform.RebaseToCliffordSingles().apply(clifford_circuit)
    for string in partition:
        transformed_string = apply_clifford_basis_change(string, clifford_circuit)
        transformed_dict = transformed_string.map
        for qubit in transformed_dict:
            if transformed_dict[qubit] == Pauli.X:
                eigenstate_circuit.H(qubit)
            if transformed_dict[qubit] == Pauli.Y:
                eigenstate_circuit.V(qubit)
        Transform.RemoveRedundancies().apply(eigenstate_circuit)
    return eigenstate_circuit


def DFSC_circuit_task_gen() -> MitTask:
    """
    For each experiment, the ansatz circuit has all symbolic gates substituted for Clifford angles (in this case, all 0's).
    If any non symbolic gates are non Clifford, an error is thrown.
    For each Clifford ansatz circuit, a new ObservableTracke is forme with new measurement circuits
    added for each Qubit Pauli String in the operator.

    :return: MitTask object that produces characterisation circuits for DFSC on a new wire as new experiments
    """

    def task(
        obj,
        measurement_wires: List[ObservableExperiment],
    ) -> Tuple[
        List[ObservableExperiment],
        List[List[List[ObservableExperiment]]],
    ]:
        """
        :param measurement_wires: A list of tuples, each tuple representing a different experiment

        :return: Original experiment wires and another list of characterisation experiments for each original experiment.
        These are organised in later task.
        """
        characterisation_wires = []
        for measurement_wire in measurement_wires:
            ansatz_circ = measurement_wire.AnsatzCircuit
            base_circ = ansatz_circ[0]
            tracker = measurement_wire.ObservableTracker
            # Given a circuit, sets all Symbols to Clifford angles
            # If circuit has non-Clifford elements not as symbolics, error thrown
            clifford_circuits = get_clifford_mcs(base_circ)

            # make a new ObservableTracker for holding characterisation circuits
            single_experiment_wires = []
            for c in clifford_circuits:
                clifford_trackers = []
                for string in tracker._qps_to_indices.keys():
                    # each characterisation circuit must have its own observable tracker
                    # this is as the 'ansatz' circuit changes for each string + Clifford combo
                    # as state preparation changes
                    new_tracker = ObservableTracker(QubitPauliOperator({string: 1}))
                    # get measurement circuit
                    measurement_circuit_info = get_basic_measurement_circuit(string)

                    para_circuit = c.get_parametric_circuit()
                    # get preparation circuit for partition, though only pass 1 string
                    prep_circuit = preparation_circuit_for_partition(
                        para_circuit, [string]
                    )
                    # add components to get characterisation circuit
                    prep_circuit.append(para_circuit)
                    new_ansatz_c = AnsatzCircuit(
                        Circuit=prep_circuit.copy(),
                        Shots=ansatz_circ[1],
                        SymbolsDict=c._symbols,
                    )
                    prep_circuit.append(measurement_circuit_info[0])
                    # add to new tracker for given Clifford circuit

                    new_tracker.add_measurement_circuit(
                        MeasurementCircuit(prep_circuit, c._symbols),
                        [measurement_circuit_info[1]],
                    )
                    clifford_trackers.append(
                        ObservableExperiment(
                            AnsatzCircuit=new_ansatz_c, ObservableTracker=new_tracker
                        )
                    )
                # a single experiment being all pauli string measurement circuits for a single Clifford Circuit
                single_experiment_wires.append(clifford_trackers)
            # single experiment wires being all observable trackers for all sampled Clifford circuits with
            # added measurement circuits for all pauli strings in given experiment
            characterisation_wires.append(single_experiment_wires)
        return (measurement_wires, characterisation_wires)

    return MitTask(_label="DFSCCircuits", _n_in_wires=1, _n_out_wires=2, _method=task)


def DFSC_collater_task_gen() -> MitTask:
    """
    For each experiment passed to MitEx, DFSC characterisation produces an ObservableTracker
    of a single Measurement Circuit for each combination of Clifford circuit produced, eigenstates preparation
    and QubitPauliString in operator, via several nested Lists.
    This task unpackages these Lists into a single List as suitable for input to MitEx objects.
    It also stores information required to produce characterisation from resulting QubitPauliOperators out of
    MitEx object.

    :return: MitTask object that collates many BackendResult objects for a single
        frame randomisation instance and converts them into a single
        BackendResult object.
    """

    def task(
        obj,
        all_characterisation_trackers: List[List[List[ObservableExperiment]]],
    ) -> Tuple[List[ObservableExperiment], List[int]]:
        """
        :param all_characterisation_trackers: Experiment wires; outer list is experiments, second outer list
        is Cliffords, inner list is qubit pauli strings.

        :return: Wire 1; All individual experiments collated into a single wire.
        Wire 2; Indexing to produce characterisation later.
        """
        organisation_indices = []
        collated_experiments = []
        for experiment_char in all_characterisation_trackers:
            # individual list is for some cliffords
            # qps is stored in output
            base_len = 0
            for cliff_ots in experiment_char:
                base_len += len(cliff_ots)
                for qps_ot in cliff_ots:
                    collated_experiments.append(qps_ot)
            organisation_indices.append(base_len)
        return (collated_experiments, organisation_indices)

    return MitTask(_label="DFSCCollation", _n_in_wires=1, _n_out_wires=2, _method=task)


def DFSC_characterisation_task_gen() -> MitTask:
    """
    Given characterisation results for all experiments, Clifford circuits and QubitPauliStrings, produces
    a characterisation result for each Experiment.

    :return: MitTask object for organising and calculating characterisation.
    """

    def task(
        obj,
        characterisation_results: List[QubitPauliOperator],
        experiment_indexing: List[int],
    ) -> Tuple[List[QubitPauliOperator]]:
        """
        :param characterisation_results: All QubitPauliOperators returned from running experiment through some MitEx object
        :param experiment_indexing: Number of characteriastion results for each experiment, used to split results up.

        :return: Collated characterisation results, one QubitPauliOperator characterisation for each experiment
        """
        split_results = []
        lower_bound = 0
        for size in experiment_indexing:
            upper_bound = lower_bound + size
            split_results.append(characterisation_results[lower_bound:upper_bound])
            lower_bound = upper_bound

        characterisation_qpos = []
        for experiment_results in split_results:
            characterisation_dict: Dict[QubitPauliString, List[float]] = dict()
            # add all expectations for each Clifford + String combo to dict
            for qpo in experiment_results:
                for string in qpo._dict:
                    if string not in characterisation_dict:
                        characterisation_dict[string] = list()
                    characterisation_dict[string].append(float(qpo._dict[string]))
            # set entry to average of values in list
            # add characterisation for DFSC to output list
            characterisation_qpos.append(
                QubitPauliOperator(
                    {
                        k: cast(Union[complex, Expr], mean(characterisation_dict[k]))
                        for k in characterisation_dict
                    }
                )
            )
        # number of characterisation qpos should match original number of experiments
        return (characterisation_qpos,)

    return MitTask(
        _label="DFSCCharacterisation", _n_in_wires=2, _n_out_wires=1, _method=task
    )


def DFSC_correction_task_gen(zero_threshold: float) -> MitTask:
    """
    For each experiment expectation, if characterisation value greater than threshold, divide experiment expectation
    by characterisation value to correct for depolarising noise.

    :param zero_threshold: Method does not correct for zero characterisation expectation values, threshold for this zero limit.

    :return: Function for DFSC correction.
    """

    def task(
        obj,
        experiment_results: List[QubitPauliOperator],
        characterisation_results: List[QubitPauliOperator],
    ) -> Tuple[List[QubitPauliOperator]]:
        """
        :param experiment_results: QubitPauliOperators corresponding to expectations for all observable experiments.
        :param characteriastion_results: QubitPauliOperators corresponding to expectations for all characterisation experiments.

        :return: Corrected expectations as QubitPauliOperator objects.
        """
        if len(experiment_results) != len(characterisation_results):
            raise ValueError(
                "{} Experiment results and {} Characterisation results: mismatch for DFSC correction.".format(
                    len(experiment_results), len(characterisation_results)
                ),
            )
        corrected_results = []
        for experiment_qpo, characterisation_qpo in zip(
            experiment_results, characterisation_results
        ):
            new_qpo = dict()
            for key in experiment_qpo._dict:
                val = characterisation_qpo._dict[key]
                if val > zero_threshold:
                    new_qpo[key] = experiment_qpo._dict[key] / val
                else:
                    new_qpo[key] = experiment_qpo._dict[key]
            corrected_results.append(QubitPauliOperator(new_qpo))
        return (corrected_results,)

    return MitTask(_label="DFSCCorrection", _n_in_wires=2, _n_out_wires=1, _method=task)


def gen_DFSC_MitEx(backend: Backend, **kwargs) -> MitEx:
    """
    Produces a MitEx object that applies DFSC characterisation to all experiment results.

    :param backend: Backend experiments are run through.
    :key experiment_mitex: MitEx object observable experiments are run through
    :key characterisation_mitex: MitEX object characteriastion experiments are run through.

    :return: MitEx object for automatic DFSC correction of circuits.
    """

    _experiment_mitex = copy(
        kwargs.get(
            "experiment_mitex",
            MitEx(
                backend,
                _label="ExperimentMitex",
                mitres=gen_compiled_MitRes(backend, 0),
            ),
        )
    )
    _characterisation_mitex = copy(
        kwargs.get(
            "characterisation_mitex",
            MitEx(
                backend,
                _label="CharacterisationMitex",
                mitres=gen_compiled_MitRes(backend, 0),
            ),
        )
    )

    _characterisation_taskgraph = TaskGraph().from_TaskGraph(_characterisation_mitex)
    _experiment_taskgraph = TaskGraph().from_TaskGraph(_experiment_mitex)

    _characterisation_taskgraph.add_wire()
    _characterisation_taskgraph.prepend(DFSC_collater_task_gen())
    _characterisation_taskgraph.append(DFSC_characterisation_task_gen())

    _experiment_taskgraph.parallel(
        MitEx(backend).from_TaskGraph(_characterisation_taskgraph)
    )
    _experiment_taskgraph.prepend(DFSC_circuit_task_gen())
    _experiment_taskgraph.append(
        DFSC_correction_task_gen(kwargs.get("DFSC_threshold", 0.01))
    )
    return MitEx(backend).from_TaskGraph(_experiment_taskgraph)

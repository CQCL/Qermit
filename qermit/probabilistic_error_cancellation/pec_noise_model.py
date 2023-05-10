from qermit import MitEx
from qermit import MitTask, ObservableExperiment
from typing import List, Tuple, Dict
from qiskit_aer.noise import NoiseModel  # type: ignore
from pytket.utils import QubitPauliOperator
from pytket.pauli import QubitPauliString  # type: ignore
from qermit import TaskGraph, AnsatzCircuit


def gen_add_pauli_gates(noise_model: NoiseModel) -> MitTask:

    def add_pauli_gates(
        obj,
        experiemnt_list: List[ObservableExperiment]
    ) -> Tuple[List[ObservableExperiment], List[Dict[str, float]], List[Dict[QubitPauliString, float]]]:

        def modify_circuit(circuit):
            return circuit

        n_circuit = 3

        experiment_info = []
        modified_experiemnt_list = []
        empty_result_list = []

        for i, experiment in enumerate(experiemnt_list):

            circuit = experiment.AnsatzCircuit.Circuit
            shots = experiment.AnsatzCircuit.Shots
            symbols_dict = experiment.AnsatzCircuit.SymbolsDict

            observable_tracker = experiment.ObservableTracker

            for _ in range(n_circuit):

                modified_circuit = modify_circuit(circuit)
                modified_ansatz_circuit = AnsatzCircuit(
                    modified_circuit, shots, symbols_dict
                )
                modified_experiemnt = ObservableExperiment(
                    modified_ansatz_circuit, observable_tracker
                )
                modified_experiemnt_list.append(modified_experiemnt)
                experiment_info.append({'experiment id': i, 'weight': 1 / n_circuit})

            observable_tracker_dict = observable_tracker._qubit_pauli_operator._dict
            observable_tracker_dict = {key: 0 for key in observable_tracker_dict.keys()}
            empty_result_list.append(observable_tracker_dict)

        return (modified_experiemnt_list, experiment_info, empty_result_list, )

    return MitTask(
        _label="AddPauliGates",
        _n_out_wires=3,
        _n_in_wires=1,
        _method=add_pauli_gates,
    )


def gen_combine_results() -> MitTask:

    def combine_results(
        obj,
        result_list: List[QubitPauliOperator],
        experiment_info: List[Dict[str, float]],
        combined_result_list: List[Dict[QubitPauliString, float]],
    ) -> Tuple[List[QubitPauliOperator]]:

        for result, info in zip(result_list, experiment_info):
            for qps, coef in result._dict.items():
                combined_result_list[info['experiment id']][qps] += coef * info['weight']  # type: ignore

        return (
            [
                QubitPauliOperator(dictionary=combined_result)  # type: ignore
                for combined_result in combined_result_list
            ],
        )

    return MitTask(
        _label="CombineResults",
        _n_out_wires=1,
        _n_in_wires=3,
        _method=combine_results,
    )


def gen_PEC_noise_model_MitEx(device_backend, noise_model):

    device_mitex = MitEx(device_backend, _label="DeviceMitex")

    pec_taskgraph = TaskGraph().from_TaskGraph(device_mitex)
    pec_taskgraph.add_wire()
    pec_taskgraph.add_wire()
    pec_taskgraph.prepend(gen_add_pauli_gates(noise_model))
    pec_taskgraph.append(gen_combine_results())

    return MitEx(device_backend).from_TaskGraph(pec_taskgraph)

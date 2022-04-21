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


from qermit import (  # type: ignore
    ObservableTracker,
    SymbolsDict,
)
from qermit.probabilistic_error_cancellation import (  # type: ignore
    gen_PEC_learning_based_MitEx,
)
from qermit.probabilistic_error_cancellation.pec_learning_based import (  # type: ignore
    gen_rebase_to_frames_and_computing,
    gen_label_gates,
    gen_wrap_frame_gates,
    gen_get_noisy_circuits,
    gen_get_clifford_training_set,
    collate_results_task_gen,
    gen_run_with_quasi_prob,
    learn_quasi_probs_task_gen,
)
from pytket.extensions.qiskit import AerBackend  # type: ignore
from pytket.pauli import Pauli, QubitPauliString  # type: ignore
from pytket.utils import QubitPauliOperator
from pytket import Circuit, Qubit, OpType
import math
from pytket.predicates import GateSetPredicate, CliffordCircuitPredicate  # type: ignore
from qermit import AnsatzCircuit, ObservableExperiment
from pytket.extensions.qiskit import IBMQEmulatorBackend
from qiskit import IBMQ  # type: ignore
import pytest

skip_remote_tests: bool = not IBMQ.stored_account()
REASON = "IBMQ account not configured"


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_no_qubit_relabel():

    noiseless_backend = AerBackend()
    lagos_backend = IBMQEmulatorBackend(
        "ibm_lagos", hub="partner-cqc", group="internal", project="default"
    )
    pec_mitex = gen_PEC_learning_based_MitEx(
        device_backend=lagos_backend, simulator_backend=noiseless_backend
    )

    c = Circuit(3)
    c.CZ(0, 2).CZ(1, 2)

    qubit_pauli_string = QubitPauliString(
        [Qubit(0), Qubit(1), Qubit(2)], [Pauli.Z, Pauli.Z, Pauli.Z]
    )
    ansatz_circuit = AnsatzCircuit(c, 2000, SymbolsDict())

    exp = [
        ObservableExperiment(
            ansatz_circuit,
            ObservableTracker(QubitPauliOperator({qubit_pauli_string: 1.0})),
        )
    ]
    result = pec_mitex.run(exp)[0]
    assert result.all_qubits == {Qubit(0), Qubit(1), Qubit(2)}


def test_gen_run_with_quasi_prob():

    noisy_prob = 0.8
    ideal_prob = 1

    task = gen_run_with_quasi_prob()
    assert task.n_in_wires == 3
    assert task.n_out_wires == 1

    wire = example_obs_exp_wire()

    num_pauli_errors = 16

    noisy_results = []
    prob_list = []
    # building arguments for tasks
    for experiment in wire:
        qpo = experiment[1].qubit_pauli_operator
        qpo_dict = {}
        qpo_prob_list = []
        for qps in qpo._dict:
            qpo_dict[qps] = noisy_prob
            qpo_prob_list.append(
                [(ideal_prob / noisy_prob) / num_pauli_errors for _ in range(16)]
            )
        prob_list.append(qpo_prob_list)
        for _ in range(num_pauli_errors):
            noisy_results.append(QubitPauliOperator(qpo_dict))

    noisy_list_structure = []
    # run test
    output = task([prob_list, noisy_results, noisy_list_structure])

    output_wire = output[0]
    for output_qpo, input_experiment in zip(output_wire, wire):

        input_qpo = input_experiment[1]

        output_qpo_dict = output_qpo._dict
        input_qpo_dict = input_qpo.qubit_pauli_operator._dict

        for output_qps, input_qps in zip(output_qpo_dict, input_qpo_dict):

            assert output_qps == input_qps
            assert math.isclose(output_qpo_dict[output_qps], ideal_prob, rel_tol=0.001)


def example_obs_exp_wire():

    circ_0 = (
        Circuit(2)
        .add_gate(OpType.noop, [1], opgroup="pre Pauli 1 0")
        .add_gate(OpType.noop, [0], opgroup="pre Pauli 1 1")
        .CX(1, 0, opgroup="Frame 1")
        .add_gate(OpType.noop, [1], opgroup="post Pauli 1 0")
        .add_gate(OpType.noop, [0], opgroup="post Pauli 1 1")
        .add_gate(OpType.U3, [0.1, 0.2, 0.3], [1], opgroup="Computing 1")
    )
    qpo_0 = QubitPauliOperator(
        {
            QubitPauliString([Qubit(0)], [Pauli.I]): 1,
            QubitPauliString([Qubit(1)], [Pauli.Z]): 1,
        }
    )
    ansatz_circ_0 = AnsatzCircuit(circ_0, 100, SymbolsDict())
    experiment_0 = ObservableExperiment(
        ansatz_circ_0,
        ObservableTracker(qpo_0),
    )

    circ_1 = (
        Circuit(2)
        .add_gate(OpType.U1, [0.3], [1], opgroup="Computing 0")
        .add_gate(OpType.noop, [0], opgroup="pre Pauli 0 0")
        .add_gate(OpType.noop, [1], opgroup="pre Pauli 0 1")
        .CX(0, 1, opgroup="Frame 0")
        .add_gate(OpType.noop, [0], opgroup="post Pauli 0 0")
        .add_gate(OpType.noop, [1], opgroup="post Pauli 0 1")
    )
    qpo_1 = QubitPauliOperator(
        {
            QubitPauliString([Qubit(0)], [Pauli.X]): 1,
            QubitPauliString([Qubit(1)], [Pauli.Y]): 1,
        }
    )
    ansatz_circ_1 = AnsatzCircuit(circ_1, 100, SymbolsDict())
    experiment_1 = ObservableExperiment(
        ansatz_circ_1,
        ObservableTracker(qpo_1),
    )

    return [experiment_0, experiment_1]


def test_collate_results_task_gen():

    noisy_prob = 0.8
    ideal_prob = 1

    num_cliff_circ = 4

    task = collate_results_task_gen()
    assert task.n_in_wires == 4
    assert task.n_out_wires == 1

    wire = example_obs_exp_wire()

    noisy_results = []
    ideal_results = []

    ideal_list_structure = []
    noisy_list_structure = []

    num_pauli_errors = 16

    clifford_exp_count = 0

    for experiment_num, experiment in enumerate(wire):
        for qps_num, qps in enumerate(
            experiment.ObservableTracker.qubit_pauli_operator._dict
        ):
            for cliff_num in range(num_cliff_circ):
                ideal_results.append(QubitPauliOperator({qps: ideal_prob}))
                ideal_list_structure.append(
                    {
                        "experiment": experiment_num,
                        "qps": qps_num,
                        "training circuit": cliff_num,
                    }
                )
                for pauli_num in range(num_pauli_errors):
                    noisy_results.append(QubitPauliOperator({qps: noisy_prob}))
                    noisy_list_structure.append(
                        {"experiment": clifford_exp_count, "error": pauli_num}
                    )
                clifford_exp_count += 1

    output = task(
        [noisy_results, noisy_list_structure, ideal_results, ideal_list_structure]
    )

    output_wire = output[0]

    for experiment_results in output_wire:
        for qps_results in experiment_results:
            for cliff_results in qps_results:
                assert len(cliff_results) == num_pauli_errors


def test_learn_quasi_probs_task_gen():

    noisy_prob = 0.8
    ideal_prob = 1

    num_cliff_circ = 4

    task = learn_quasi_probs_task_gen(num_cliff_circ=num_cliff_circ)

    assert task.n_in_wires == 1
    assert task.n_out_wires == 1

    wire = example_obs_exp_wire()

    num_pauli_errors = 16

    results = []

    for experiment_num, experiment in enumerate(wire):
        fixed_exp_results = []
        for qps_num, qps in enumerate(
            experiment.ObservableTracker.qubit_pauli_operator._dict
        ):
            fixed_qps_results = []
            for cliff_num in range(num_cliff_circ):
                fixed_cliff_results = []
                for pauli_num in range(num_pauli_errors):
                    fixed_cliff_results.append(
                        (
                            QubitPauliOperator({qps: noisy_prob}),
                            QubitPauliOperator({qps: ideal_prob}),
                        )
                    )
                fixed_qps_results.append(fixed_cliff_results)
            fixed_exp_results.append(fixed_qps_results)
        results.append(fixed_exp_results)

    output = task([results])

    output_wire = output[0]

    for experiment_probs in output_wire:
        for qps_probs in experiment_probs:
            assert len(qps_probs) == num_pauli_errors
            assert math.isclose(sum(qps_probs) * noisy_prob, ideal_prob, rel_tol=0.001)

    return


def test_gen_get_clifford_training_set():

    be = AerBackend()

    num_rand_cliff = 4

    task = gen_get_clifford_training_set(
        simulator_backend=be, num_rand_cliff=num_rand_cliff
    )
    assert task.n_in_wires == 1
    assert task.n_out_wires == 2

    qpo = QubitPauliOperator(
        {QubitPauliString([Qubit(0), Qubit(1)], [Pauli.Z, Pauli.Z]): 1}
    )

    circ = (
        Circuit(2)
        .add_gate(OpType.U3, [0.1, 0.2, 0.3], [0], opgroup="Computing 0")
        .add_gate(OpType.U3, [0.1, 0.2, 0.3], [1], opgroup="Computing 1")
        .CX(0, 1, opgroup="Frame 1")
    )
    ansatz_circ = AnsatzCircuit(circ, 1000, SymbolsDict)
    experiment = ObservableExperiment(ansatz_circ, ObservableTracker(qpo))
    wire = [experiment]

    output = task([wire])
    output_wire = output[0]
    assert len(output_wire) == num_rand_cliff * len(wire)

    input_wire_expanded = [
        input_experiment for input_experiment in wire for _ in range(num_rand_cliff)
    ]

    for input_experiment, output_experiment in zip(input_wire_expanded, output_wire):

        input_circuit = input_experiment[0][0]
        output_circuit = output_experiment[0][0]

        # Check that the output circuit is Clifford circuit
        assert CliffordCircuitPredicate().verify(output_circuit)

        # Check that all Computing gates have been replaced with Clifford gates
        for input_circuit_command, output_circuit_command in zip(
            input_circuit.get_commands(), output_circuit.get_commands()
        ):
            if "Computing" in input_circuit_command.opgroup:
                assert "Clifford" in output_circuit_command.opgroup


def test_gen_get_noisy_circuits():

    be = AerBackend()

    task = gen_get_noisy_circuits(backend=be)
    assert task.n_in_wires == 1
    assert task.n_out_wires == 2

    # Building two circuits that are labled
    circ_0 = (
        Circuit(2)
        .add_gate(OpType.noop, [1], opgroup="pre Pauli 1 0")
        .add_gate(OpType.noop, [0], opgroup="pre Pauli 1 1")
        .CX(1, 0, opgroup="Frame 1")
        .add_gate(OpType.noop, [1], opgroup="post Pauli 1 0")
        .add_gate(OpType.noop, [0], opgroup="post Pauli 1 1")
        .add_gate(OpType.U3, [0.1, 0.2, 0.3], [1], opgroup="Computing 1")
    )
    ansatz_circ_0 = AnsatzCircuit(circ_0, 100, SymbolsDict())
    experiment_0 = ObservableExperiment(
        ansatz_circ_0, ObservableTracker(QubitPauliOperator())
    )

    circ_1 = (
        Circuit(2)
        .add_gate(OpType.U1, [0.3], [1], opgroup="Computing 0")
        .add_gate(OpType.noop, [0], opgroup="pre Pauli 0 0")
        .add_gate(OpType.noop, [1], opgroup="pre Pauli 0 1")
        .CX(0, 1, opgroup="Frame 0")
        .add_gate(OpType.noop, [0], opgroup="post Pauli 0 0")
        .add_gate(OpType.noop, [1], opgroup="post Pauli 0 1")
    )
    ansatz_circ_1 = AnsatzCircuit(circ_1, 100, SymbolsDict())
    experiment_1 = ObservableExperiment(
        ansatz_circ_1, ObservableTracker(QubitPauliOperator())
    )

    wire = [experiment_0, experiment_1]

    # Testing that the output is of the correct form
    output = task([wire])
    assert len(output) == 2

    # There should be 15 noisy circuits per Frame gate, +1 noise free circuit
    output_wire = output[0]
    assert len(output_wire) == len(wire) * 16

    # Check that the first noisy circuit is identical to the original (i.e. contains no noise)
    assert output_wire[0][0][0] == circ_0
    assert output_wire[16][0][0] == circ_1

    input_wire_expanded = [experiment for experiment in wire for _ in range(16)]

    for input_experiment, output_experiment in zip(input_wire_expanded, output_wire):

        input_command_list = input_experiment[0][0].get_commands()
        output_command_list = output_experiment[0][0].get_commands()

        for input_command, output_command in zip(
            input_command_list, output_command_list
        ):

            # Check that all gates that are not Pauli gates are the same
            if ("Frame" in input_command.opgroup) or (
                "Computing" in input_command.opgroup
            ):

                assert input_command.opgroup == output_command.opgroup

    # Check the correctness of an instances
    output_circ = output_wire[7][0][0]

    output_circ_commands = output_circ.get_commands()

    for command in output_circ_commands:

        if command.opgroup == "pre Pauli 1 0":
            assert command.op.get_name() == "Y"
        elif command.opgroup == "pre Pauli 1 1":
            assert command.op.get_name() == "Z"
        elif command.opgroup == "post Pauli 1 0":
            assert command.op.get_name() == "Y"
        elif command.opgroup == "post Pauli 1 1":
            assert command.op.get_name() == "Z"


def test_gen_wrap_frame_gates():

    task = gen_wrap_frame_gates()
    assert task.n_in_wires == 1
    assert task.n_out_wires == 1

    # Building two circuits that are labled
    circ_0 = (
        Circuit(2)
        .CX(1, 0, opgroup="Frame 1")
        .add_gate(OpType.U3, [0.1, 0.2, 0.3], [1], opgroup="Computing 1")
    )
    ansatz_circ_0 = AnsatzCircuit(circ_0, 100, SymbolsDict())
    experiment_0 = ObservableExperiment(
        ansatz_circ_0, ObservableTracker(QubitPauliOperator())
    )

    circ_1 = (
        Circuit(2)
        .add_gate(OpType.U1, [0.3], [1], opgroup="Computing 0")
        .CX(0, 1, opgroup="Frame 0")
    )
    ansatz_circ_1 = AnsatzCircuit(circ_1, 100, SymbolsDict())
    experiment_1 = ObservableExperiment(
        ansatz_circ_1, ObservableTracker(QubitPauliOperator())
    )

    wire = [experiment_0, experiment_1]

    # Testing that the output is of the correct form
    output = task([wire])
    assert len(output) == 1

    output_wire = output[0]
    assert len(output_wire) == len(wire)

    output_circ_0 = output_wire[0][0][0]
    output_circ_1 = output_wire[1][0][0]

    output_circ_0_commands = output_circ_0.get_commands()
    output_circ_1_commands = output_circ_1.get_commands()

    # Testing that the gates are added and labelled correctly
    assert output_circ_0_commands[0].opgroup == "pre Pauli 1 1"
    assert output_circ_0_commands[1].opgroup == "pre Pauli 1 0"
    assert output_circ_0_commands[2].opgroup == "Frame 1"
    assert output_circ_0_commands[3].opgroup == "post Pauli 1 1"
    assert output_circ_0_commands[4].opgroup == "post Pauli 1 0"
    assert output_circ_0_commands[5].opgroup == "Computing 1"

    assert output_circ_1_commands[0].opgroup == "pre Pauli 0 0"
    assert output_circ_1_commands[1].opgroup == "Computing 0"
    assert output_circ_1_commands[2].opgroup == "pre Pauli 0 1"
    assert output_circ_1_commands[3].opgroup == "Frame 0"
    assert output_circ_1_commands[4].opgroup == "post Pauli 0 0"
    assert output_circ_1_commands[5].opgroup == "post Pauli 0 1"


def test_gen_label_gates():

    task = gen_label_gates()

    circ_0 = (
        Circuit(2)
        .CX(1, 0)
        .add_gate(OpType.TK1, [0.1, 0.2, 0.3], [1])
        .add_gate(OpType.TK1, [0.2, 0.3, 0], [0])
    )
    ansatz_circ_0 = AnsatzCircuit(circ_0, 100, SymbolsDict())
    experiment_0 = ObservableExperiment(
        ansatz_circ_0, ObservableTracker(QubitPauliOperator())
    )

    circ_1 = (
        Circuit(2)
        .add_gate(OpType.TK1, [0.3, 0, 0], [1])
        .CX(0, 1)
        .add_gate(OpType.TK1, [0.2, 0.3, 0], [1])
    )
    ansatz_circ_1 = AnsatzCircuit(circ_1, 100, SymbolsDict())
    experiment_1 = ObservableExperiment(
        ansatz_circ_1, ObservableTracker(QubitPauliOperator())
    )

    wire = [experiment_0, experiment_1]

    output = task([wire])
    assert len(output) == 1

    output_wire = output[0]
    assert len(output_wire) == len(wire)

    for output_experiment, input_experiment in zip(output_wire, wire):
        output_circuit = output_experiment[0][0]
        input_circuit = input_experiment[0][0]

        assert len(output_circuit.get_commands()) == len(input_circuit.get_commands())

        output_circuit_dict = output_circuit.to_dict()
        input_circuit_dict = input_circuit.to_dict()

        for input_command, output_command in zip(
            input_circuit_dict["commands"], output_circuit_dict["commands"]
        ):
            assert input_command["args"] == output_command["args"]
            assert input_command["op"] == output_command["op"]

    output_circ_0 = output_wire[0][0][0]
    output_circ_1 = output_wire[1][0][0]

    output_circ_0_commands = output_circ_0.get_commands()
    output_circ_1_commands = output_circ_1.get_commands()

    # Testing that the gates are indeed correctly labled
    assert output_circ_0_commands[0].opgroup == "Frame 0"
    assert output_circ_0_commands[1].opgroup == "Computing 0"
    assert output_circ_0_commands[2].opgroup == "Computing 1"

    assert output_circ_1_commands[0].opgroup == "Computing 0"
    assert output_circ_1_commands[1].opgroup == "Frame 0"
    assert output_circ_1_commands[2].opgroup == "Computing 1"


def test_gen_rebase_to_frames_and_computing():

    task = gen_rebase_to_frames_and_computing()

    circ_0 = Circuit(2).CX(1, 0).X(0).Rx(0.25, 1)
    ansatz_circ_0 = AnsatzCircuit(circ_0, 100, SymbolsDict())
    experiment_0 = ObservableExperiment(
        ansatz_circ_0, ObservableTracker(QubitPauliOperator())
    )

    circ_1 = Circuit(2).Y(1).CZ(1, 0).Ry(0.25, 1)
    ansatz_circ_1 = AnsatzCircuit(circ_1, 100, SymbolsDict())
    experiment_1 = ObservableExperiment(
        ansatz_circ_1, ObservableTracker(QubitPauliOperator())
    )

    wire = [experiment_0, experiment_1]

    output = task([wire])
    assert len(output) == 1

    output_wire = output[0]
    assert len(output_wire) == len(wire)

    for output_experiment in output_wire:
        circuit = output_experiment[0][0]
        assert GateSetPredicate({OpType.CX, OpType.TK1}).verify(circuit)


def test_gen_PEC_learning_based_MitEx():

    be = AerBackend()

    me = gen_PEC_learning_based_MitEx(
        device_backend=be,
        simulator_backend=be,
        _label="TestPECMitEx",
        optimisation_level=0,
        num_cliff=20,
    )

    qpo_Z = QubitPauliOperator({QubitPauliString([Qubit(0)], [Pauli.Z]): 1})
    qpo_ZZ = QubitPauliOperator(
        {
            QubitPauliString([Qubit(0)], [Pauli.Z]): 1,
            QubitPauliString([Qubit(1)], [Pauli.Z]): 1,
        }
    )

    c = Circuit(2).CZ(0, 1).T(1)
    ac = AnsatzCircuit(c, 1000, SymbolsDict())

    circ_list = [
        ObservableExperiment(ac, ObservableTracker(qpo_Z)),
        ObservableExperiment(ac, ObservableTracker(qpo_ZZ)),
    ]
    result_list = me.run(circ_list)
    assert len(result_list) == 2

    for result in result_list:
        result_dict = result._dict
        for qps in result_dict:
            assert math.isclose(result_dict[qps], 1, rel_tol=0.001)


if __name__ == "__main__":
    test_no_qubit_relabel()
    test_gen_run_with_quasi_prob()
    test_collate_results_task_gen()
    test_learn_quasi_probs_task_gen()
    test_gen_get_clifford_training_set()
    test_gen_get_noisy_circuits()
    test_gen_wrap_frame_gates()
    test_gen_label_gates()
    test_gen_rebase_to_frames_and_computing()
    test_gen_PEC_learning_based_MitEx()

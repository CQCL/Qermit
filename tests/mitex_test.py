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
    MitEx,
    SymbolsDict,
    ObservableTracker,
    CircuitShots,
    AnsatzCircuit,
)
from qermit.taskgraph.mitex import (  # type: ignore
    filter_observable_tracker_task_gen,
    collate_circuit_shots_task_gen,
    split_results_task_gen,
    get_expectations_task_gen,
    gen_compiled_shot_split_MitRes,
)
import copy
from pytket.circuit import Circuit, fresh_symbol, Qubit, OpType  # type: ignore
from pytket.pauli import QubitPauliString, Pauli  # type: ignore
from pytket.utils import QubitPauliOperator
from pytket.extensions.qiskit import AerBackend  # type: ignore


def gen_test_wire_objs():
    sym_0 = fresh_symbol("alpha")
    sym_1 = fresh_symbol("beta")
    c = Circuit(5).CX(0, 1).CX(1, 2).Rz(sym_0, 2).CX(1, 2).CX(0, 1)
    c.X(0).X(3).CX(0, 3).CX(3, 4).Rz(sym_1, 4).CX(3, 4).CX(0, 3)
    sd0 = SymbolsDict.symbols_from_dict({sym_0: 0.5, sym_1: 1})
    sd1 = SymbolsDict.symbols_from_dict({sym_0: 0.4, sym_1: 0.6})

    ansatz_circuit0 = AnsatzCircuit(c.copy(), 50, sd0)
    ansatz_circuit1 = AnsatzCircuit(c.copy(), 60, sd1)

    # make observable tracker
    qps_0 = QubitPauliString(
        [Qubit(0), Qubit(1), Qubit(2)], [Pauli.X, Pauli.Y, Pauli.Z]
    )
    qps_1 = QubitPauliString(
        [Qubit(3), Qubit(1), Qubit(4)], [Pauli.Y, Pauli.Z, Pauli.Z]
    )
    operator0 = QubitPauliOperator(
        {
            copy.copy(qps_0): 1.0,
            copy.copy(qps_1): 1.0,
        }
    )
    tracker0 = ObservableTracker(operator0)
    tracker1 = ObservableTracker(operator0)
    return (ansatz_circuit0, ansatz_circuit1, tracker0, tracker1)


def test_filter_observable_tracker_task_gen():
    ansatz_circuit0, ansatz_circuit1, observable0, observable1 = gen_test_wire_objs()
    # make measurement wires
    # make ansatz circuit
    task = filter_observable_tracker_task_gen()
    assert task.n_in_wires == 1
    assert task.n_out_wires == 2

    wires = [(ansatz_circuit0, observable0), (ansatz_circuit1, observable1)]
    output = task([wires])

    # take [0][0] as comes as List[List[CircuitShots]]
    circuits = output[0]
    assert len(circuits) == 2

    circuits0 = circuits[0]
    circuits1 = circuits[1]

    trackers = output[1]
    assert len(trackers) == 2
    trackers0 = trackers[0]
    trackers1 = trackers[1]

    assert circuits0[0][1] == 50
    assert circuits0[1][1] == 50
    assert circuits1[0][1] == 60
    assert circuits1[1][1] == 60

    # both circuits have same base length so can use for both
    base_commands_length = len(ansatz_circuit0[0].get_commands())
    # corresponds to basis change and measures

    c00_commands = circuits0[0][0].get_commands()
    c01_commands = circuits0[1][0].get_commands()
    c10_commands = circuits1[0][0].get_commands()
    c11_commands = circuits1[1][0].get_commands()

    assert len(c00_commands) == base_commands_length + 5
    assert len(c01_commands) == base_commands_length + 4
    assert len(c10_commands) == base_commands_length + 5
    assert len(c11_commands) == base_commands_length + 4

    # check symbolics are substituted correctly
    assert c00_commands[3].op.params[0] == 0.5
    assert c00_commands[12].op.params[0] == 1.0
    assert c01_commands[3].op.params[0] == 0.5
    assert c01_commands[10].op.params[0] == 1.0
    assert c10_commands[3].op.params[0] == 0.4
    assert c10_commands[12].op.params[0] == 0.6
    assert c11_commands[3].op.params[0] == 0.4
    assert c11_commands[10].op.params[0] == 0.6

    # assert measures exist
    assert c00_commands[5].op.type == OpType.Measure
    assert c00_commands[9].op.type == OpType.Measure
    assert c00_commands[16].op.type == OpType.Measure
    assert c01_commands[6].op.type == OpType.Measure
    assert c01_commands[12].op.type == OpType.Measure
    assert c01_commands[15].op.type == OpType.Measure
    assert c10_commands[5].op.type == OpType.Measure
    assert c10_commands[9].op.type == OpType.Measure
    assert c10_commands[16].op.type == OpType.Measure
    assert c11_commands[6].op.type == OpType.Measure
    assert c11_commands[12].op.type == OpType.Measure
    assert c11_commands[15].op.type == OpType.Measure

    # check tracker objects
    assert len(trackers0.measurement_circuits) == 2
    assert len(trackers1.measurement_circuits) == 2
    assert len(trackers0.get_empty_strings()) == 0
    assert len(trackers1.get_empty_strings()) == 0


def test_collate_and_split_circuit_shots_task_gen():
    ansatz_circuit0, ansatz_circuit1, observable0, observable1 = gen_test_wire_objs()
    filter_task = filter_observable_tracker_task_gen()
    wires = [(ansatz_circuit0, observable0), (ansatz_circuit1, observable1)]

    filter_outputs = filter_task([wires])
    original_circuits = filter_outputs[0]

    # as weve tested filter task, use to set arguments up...
    collate_task = collate_circuit_shots_task_gen()
    assert collate_task.n_in_wires == 1
    assert collate_task.n_out_wires == 2

    # [0] to unpack tuple
    split_output = collate_task([original_circuits])
    assert len(split_output) == 2
    # output 1 gives the number of circuits for each experiment
    assert [2, 2] == split_output[1]

    circuits = split_output[0]
    assert len(circuits) == 4

    # simulate results
    backend = AerBackend()
    just_circuits = []
    for c in circuits:
        just_circuits.append(backend.get_compiled_circuit(c[0]))
    handles = backend.process_circuits(just_circuits, 1)
    results = backend.get_results(handles)

    # check split_results_task
    split_results_task = split_results_task_gen()
    assert split_results_task.n_in_wires == 2
    assert split_results_task.n_out_wires == 1

    recollated_output = split_results_task([results, split_output[1]])[0]

    assert len(recollated_output) == 2
    assert len(recollated_output[0]) == 2
    assert len(recollated_output[1]) == 2


def test_get_expectations_task_gen():
    ansatz_circuit0, ansatz_circuit1, observable0, observable1 = gen_test_wire_objs()
    filter_task = filter_observable_tracker_task_gen()
    wires = [(ansatz_circuit0, observable0), (ansatz_circuit1, observable1)]
    # as weve tested filter task, collate task and splitter task,
    # use to set get_xpectations test...
    filter_outputs = filter_task([wires])
    original_circuits = filter_outputs[0]
    trackers = filter_outputs[1]

    collate_task = collate_circuit_shots_task_gen()
    # [0] to unpack tuple
    split_output = collate_task([original_circuits])
    # simulate results
    backend = AerBackend()
    just_circuits = [backend.get_compiled_circuit(c[0]) for c in split_output[0]]
    handles = backend.process_circuits(just_circuits, 5)
    results = backend.get_results(handles)

    # check split_results_task
    split_results_task = split_results_task_gen()
    recollated_output = split_results_task([results, split_output[1]])[0]

    expectation_wire = [recollated_output, trackers]
    expectations_task = get_expectations_task_gen()
    assert expectations_task.n_in_wires == 2
    assert expectations_task.n_out_wires == 1

    qpos = expectations_task(expectation_wire)[0]
    assert len(qpos) == 2
    assert len(qpos[0]._dict) == 2
    assert len(qpos[1]._dict) == 2
    values_0 = list(qpos[0]._dict.values())
    values_1 = list(qpos[1]._dict.values())
    assert values_0[0] >= -1
    assert values_0[0] <= 1
    assert values_0[1] >= -1
    assert values_0[1] <= 1
    assert values_1[0] >= -1
    assert values_1[0] <= 1
    assert values_1[1] >= -1
    assert values_1[1] <= 1


# test specific MitEx methods
def test_mitex_run():
    # create ansatz circuit objefts
    c0 = AnsatzCircuit(Circuit(3).X(0).X(1), 10, SymbolsDict())
    c1 = AnsatzCircuit(Circuit(3).X(1).X(2), 10, SymbolsDict())
    # create operator stirngs
    qps_12 = QubitPauliString([Qubit(1), Qubit(2)], [Pauli.Z, Pauli.Z])
    qps_01 = QubitPauliString([Qubit(0), Qubit(1)], [Pauli.Z, Pauli.Z])
    qps_012 = QubitPauliString(
        [Qubit(0), Qubit(1), Qubit(2)], [Pauli.Z, Pauli.Z, Pauli.Z]
    )
    # create observable tracker objects
    t0 = ObservableTracker(QubitPauliOperator({qps_01: 0.5, qps_12: 1.0}))
    t1 = ObservableTracker(QubitPauliOperator({qps_012: 0.7}))
    # run experiments
    experiment = [(c0, t0), (c1, t1)]
    me = MitEx(AerBackend())
    res = me.run(experiment)
    assert len(res) == 2
    assert res[0][qps_01] == 0.5
    assert res[0][qps_12] == -1.0
    assert res[1][qps_012] == 0.7


def test_mitex_run_basic():
    # create ansatz circuit objects
    c0 = CircuitShots(Circuit(3).X(0).X(1), 10)
    c1 = CircuitShots(Circuit(3).X(1).X(2), 10)
    # create operator stirngs
    qps_12 = QubitPauliString([Qubit(1), Qubit(2)], [Pauli.Z, Pauli.Z])
    qps_01 = QubitPauliString([Qubit(0), Qubit(1)], [Pauli.Z, Pauli.Z])
    qps_012 = QubitPauliString(
        [Qubit(0), Qubit(1), Qubit(2)], [Pauli.Z, Pauli.Z, Pauli.Z]
    )
    # create observable tracker objects
    qpo0 = QubitPauliOperator({qps_01: 0.5, qps_12: 1.0})
    qpo1 = QubitPauliOperator({qps_012: 0.7})
    # run experiments
    experiment = [(c0, qpo0), (c1, qpo1)]
    me = MitEx(AerBackend())
    res = me.run_basic(experiment)
    assert len(res) == 2
    assert res[0][qps_01] == 0.5
    assert res[0][qps_12] == -1.0
    assert res[1][qps_012] == 0.7


def test_gen_compiled_shot_split_MitRes():

    backend = AerBackend()

    mitres = gen_compiled_shot_split_MitRes(backend, 5, optimisation_level=2)
    mitres.get_task_graph()

    n_shots_1 = 8
    circ_1 = Circuit(1).X(0).X(0).measure_all()
    n_shots_2 = 12
    circ_2 = Circuit(2).CX(0, 1).measure_all()

    results = mitres.run(
        [CircuitShots(circ_1, n_shots_1), CircuitShots(circ_2, n_shots_2)]
    )

    assert len(results[0].get_shots()) == n_shots_1
    assert len(results[1].get_shots()) == n_shots_2


if __name__ == "__main__":

    # calling test methods
    test_filter_observable_tracker_task_gen()
    test_collate_and_split_circuit_shots_task_gen()
    test_get_expectations_task_gen()
    test_mitex_run()
    test_mitex_run_basic()
    test_gen_compiled_shot_split_MitRes()

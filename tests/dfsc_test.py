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


from pytket import Circuit, Qubit
from pytket.circuit import fresh_symbol  # type: ignore
from pytket.transform import Transform  # type: ignore

from qermit import (  # type: ignore
    SymbolsDict,
    ObservableTracker,
    MeasurementCircuit,
    MitEx,
    ObservableExperiment,
)
from qermit.taskgraph.mitex import get_basic_measurement_circuit  # type: ignore
from qermit.clifford_noise_characterisation import (  # type: ignore
    gen_DFSC_MitEx,
)
from qermit.clifford_noise_characterisation.dfsc import (  # type: ignore
    get_clifford_mcs,
    preparation_circuit_for_partition,
    DFSC_circuit_task_gen,
    DFSC_collater_task_gen,
    DFSC_characterisation_task_gen,
    DFSC_correction_task_gen,
)
import copy
from pytket.pauli import Pauli, QubitPauliString  # type: ignore
from pytket.utils import QubitPauliOperator
from pytket.extensions.qiskit import AerBackend  # type: ignore


sym_0 = fresh_symbol("alpha")
sym_1 = fresh_symbol("beta")
c = Circuit(5).CX(0, 1).CX(1, 2).Rz(sym_0, 2).CX(1, 2).CX(0, 1)
c.X(0).X(3).CX(0, 3).CX(3, 4).Rz(sym_1, 4).CX(3, 4).CX(0, 3)
sd0 = SymbolsDict.symbols_from_dict({sym_0: 0.5, sym_1: 1})
sd1 = SymbolsDict.symbols_from_dict({sym_0: 0.4, sym_1: 0.6})

ansatz_circuit0 = (c.copy(), 50, sd0)
ansatz_circuit1 = (c.copy(), 60, sd1)

# make observable tracker
qps_0 = QubitPauliString([Qubit(0), Qubit(1), Qubit(2)], [Pauli.X, Pauli.Y, Pauli.Z])
qps_1 = QubitPauliString([Qubit(3), Qubit(1), Qubit(4)], [Pauli.Y, Pauli.Z, Pauli.Z])
operator0 = QubitPauliOperator(
    {
        copy.copy(qps_0): 1.0,
        copy.copy(qps_1): 1.0,
    }
)
operator1 = QubitPauliOperator(
    {
        copy.copy(qps_0): 0.8,
        copy.copy(qps_1): 0.8,
    }
)
tracker0 = ObservableTracker(operator0)
tracker1 = ObservableTracker(operator0)


def test_get_clifford_mcs():
    test_circuit = Circuit(4)
    comparison_dict = dict()
    for i in range(4):
        sym = fresh_symbol("a" + str(i))
        test_circuit.Ry(sym, i)
        comparison_dict[sym] = 0
    measurement_circuit = get_clifford_mcs(test_circuit)[0]
    assert measurement_circuit._symbolic_circuit == test_circuit
    assert measurement_circuit._symbols._symbolic_map == comparison_dict


def test_prep_circuit_for_partition():
    n_qubits = 4
    test_circuit_0 = Circuit(n_qubits)
    comparison_circuit_0 = Circuit(n_qubits).H(0).V(1).H(2)

    test_circuit_1 = Circuit(n_qubits).H(0).H(1).H(2).H(3).CX(0, 1).CX(1, 2).CX(2, 3)
    comparison_circuit_1 = Circuit(n_qubits).V(0).H(1)

    qps = QubitPauliString([Qubit(0), Qubit(1), Qubit(2)], [Pauli.X, Pauli.Y, Pauli.X])
    prep_circuit_0 = preparation_circuit_for_partition(test_circuit_0, [qps])
    assert prep_circuit_0 == comparison_circuit_0

    prep_circuit_1 = preparation_circuit_for_partition(test_circuit_1, [qps])
    assert prep_circuit_1 == comparison_circuit_1


def test_DFSC_circuit_task_gen():
    # make task, assert basic MitTask properties are retained
    wires = [ObservableExperiment(ansatz_circuit0, tracker0)]
    task = DFSC_circuit_task_gen()
    assert task.n_in_wires == 1
    assert task.n_out_wires == 2

    # assert that correct number of wires are returned
    # assert first wire returns original experiment wires
    res = task([wires])
    assert len(res) == 2
    assert res[0] == wires

    characterisation_trackers = res[1]
    assert len(characterisation_trackers) == 1

    # trackers for each experiment
    # comprised of lists of ObservableTracker, where each list is for some different Clifford circuit
    exp_0_cliff_trackers = characterisation_trackers[0]

    assert len(exp_0_cliff_trackers) == 1

    # each qubit pauli string in experiment operator has a different observable tracker
    # for each Clifford circuit used for characterisation
    # this is because ansatz circuit changes for each characteriastion as state prep
    # is important
    exp_0_qps_trackers = exp_0_cliff_trackers[0]

    assert len(exp_0_qps_trackers) == 2

    #  check that correct characteriastion circuits are made for experiment 0
    exp_0_circ = wires[0][0][0]
    exp_0_qps = list(wires[0][1].qubit_pauli_operator._dict.keys())
    exp_0_cliff = get_clifford_mcs(exp_0_circ)[0].get_parametric_circuit()
    Transform.RebaseToCliffordSingles().apply(exp_0_cliff)

    # prepare state prep circuit for both qubit pauli strings in operator
    prep_circuit_00 = preparation_circuit_for_partition(exp_0_cliff, [exp_0_qps[0]])
    prep_circuit_01 = preparation_circuit_for_partition(exp_0_cliff, [exp_0_qps[1]])
    # append clifford circuit for each state prep
    prep_circuit_00.append(exp_0_cliff)
    prep_circuit_01.append(exp_0_cliff)

    # recover Clifford circuit via creating measurement circuit and substituting symbols
    # deal with horrible nested lists
    char_circuit_00 = exp_0_qps_trackers[0][0][0]
    char_symbols_00 = exp_0_qps_trackers[0][0][2]
    char_mc_00 = MeasurementCircuit(
        char_circuit_00, char_symbols_00
    ).get_parametric_circuit()
    Transform.RebaseToCliffordSingles().apply(char_mc_00)

    char_circuit_01 = exp_0_qps_trackers[1][0][0]
    char_symbols_01 = exp_0_qps_trackers[1][0][2]
    char_mc_01 = MeasurementCircuit(
        char_circuit_01, char_symbols_01
    ).get_parametric_circuit()
    Transform.RebaseToCliffordSingles().apply(char_mc_01)

    # confirm returned characteriastion state prep circuits are equal to manually created ones
    assert char_mc_00 == prep_circuit_00
    assert char_mc_01 == prep_circuit_01

    # get ObservableTracker object for each characterisation
    ot_00 = exp_0_qps_trackers[0][1]
    ot_01 = exp_0_qps_trackers[1][1]
    # check that operators in observable tracker correspond to a single string in experiment operator
    assert list(ot_00.qubit_pauli_operator._dict.keys()) == [exp_0_qps[0]]
    assert list(ot_01.qubit_pauli_operator._dict.keys()) == [exp_0_qps[1]]

    # get measurement circuits, check there is only one for the single string
    ot_00_mcs = ot_00.measurement_circuits
    ot_01_mcs = ot_01.measurement_circuits
    assert len(ot_00_mcs) == 1
    assert len(ot_01_mcs) == 1

    # get measurement circuits for each qubit pauli string in operator
    qpos_00_mc = get_basic_measurement_circuit(exp_0_qps[0])
    qpos_01_mc = get_basic_measurement_circuit(exp_0_qps[1])

    # make new prep + clifford + measurement circuits for comparison
    prep_circuit_00.append(qpos_00_mc[0])
    prep_circuit_01.append(qpos_01_mc[0])

    # get circuit object corresponding to measurement
    ot_00_para_mc = ot_00_mcs[0].get_parametric_circuit()
    ot_01_para_mc = ot_01_mcs[0].get_parametric_circuit()

    # confirm correct circuits constructed in method
    assert prep_circuit_00 == ot_00_para_mc
    assert prep_circuit_01 == ot_01_para_mc


def test_DFSC_collater_task_gen():
    wires = [
        ObservableExperiment(ansatz_circuit0, tracker0),
        ObservableExperiment(ansatz_circuit1, tracker1),
    ]
    # used this safely as tested to get data
    collate_arguments = DFSC_circuit_task_gen()([wires])[1]

    assert len(collate_arguments) == 2
    task = DFSC_collater_task_gen()
    assert task.n_in_wires == 1
    assert task.n_out_wires == 2

    collated_trackers = task([collate_arguments])
    # 2 wires out
    assert len(collated_trackers) == 2
    # one Clifford circuit for each Qubit Pauli String in Experiment Operator
    assert len(collated_trackers[0]) == 4
    # Number of characterisation circuits for each experiment
    # Each experiment has two Qubit Pauli Strings and one generated Clifford circuit i.e. 2 each
    assert collated_trackers[1] == [2, 2]


def test_DFSC_characterisation_task_gen():
    qps0 = qps_0
    qps1 = qps_1
    wires = [
        ObservableExperiment(ansatz_circuit0, tracker0),
        ObservableExperiment(ansatz_circuit1, tracker1),
    ]
    collate_args = DFSC_circuit_task_gen()([wires])[1]
    collated_res = DFSC_collater_task_gen()([collate_args])
    wires = collated_res[0]
    indexing = collated_res[1]
    me = MitEx(AerBackend())

    results = me.run(wires)
    # each characteriastion circuit should return some result
    assert len(results) == 4
    # assert each result is correct
    assert results[0]._dict[qps0] == -1
    assert results[1]._dict[qps1] == 1
    assert results[2]._dict[qps0] == -1
    assert results[3]._dict[qps1] == 1

    test_task = DFSC_characterisation_task_gen()
    assert test_task.n_in_wires == 2
    assert test_task.n_out_wires == 1
    characterisation_results = test_task([results, indexing])
    # check only 1 wire is returned
    assert len(characterisation_results) == 1
    assert len(characterisation_results[0]) == 2

    # check results are suitably recollated
    assert characterisation_results[0][0]._dict[qps0] == -1
    assert characterisation_results[0][0]._dict[qps1] == 1
    assert characterisation_results[0][1]._dict[qps0] == -1
    assert characterisation_results[0][1]._dict[qps1] == 1


def test_DFSC_correction_task_gen():
    task = DFSC_correction_task_gen(zero_threshold=0.01)
    assert task.n_in_wires == 2
    assert task.n_out_wires == 1
    # set up dummy experiment and characterisation results
    qps_0 = QubitPauliString(
        [Qubit(0), Qubit(1), Qubit(2)], [Pauli.X, Pauli.Y, Pauli.Z]
    )
    qps_1 = QubitPauliString(
        [Qubit(0), Qubit(1), Qubit(2)], [Pauli.Z, Pauli.X, Pauli.Y]
    )
    qps_2 = QubitPauliString(
        [Qubit(0), Qubit(1), Qubit(2)], [Pauli.Y, Pauli.Z, Pauli.X]
    )
    qpo_0 = QubitPauliOperator({qps_0: 2.0, qps_1: 2.0, qps_2: 2.0})
    qpo_1 = QubitPauliOperator({qps_0: 0.0, qps_1: 0.1, qps_2: 1.0})
    qpo_2 = QubitPauliOperator({qps_0: 3.0, qps_1: 3.0, qps_2: 3.0})
    qpo_3 = QubitPauliOperator({qps_0: 0.1, qps_1: 0.0, qps_2: 3.0})

    dummy_experiment_res = [qpo_0, qpo_2]
    dummy_characterisation_res = [qpo_1, qpo_3]

    dummy_corrected_res_tup = task([dummy_experiment_res, dummy_characterisation_res])

    assert len(dummy_corrected_res_tup) == 1
    dummy_corrected_res = dummy_corrected_res_tup[0]

    # assert that each correction is correct, and that characteriastion under zero threshold don't correct
    assert dummy_corrected_res[0][qps_0] == 2.0
    assert dummy_corrected_res[0][qps_1] == 20.0
    assert dummy_corrected_res[0][qps_2] == 2.0
    assert dummy_corrected_res[1][qps_0] == 30.0
    assert dummy_corrected_res[1][qps_1] == 3.0
    assert dummy_corrected_res[1][qps_2] == 1.0


def test_DFSC_mitex_gen():
    # set up experiment with guaranteed expectations
    # confirm that despite dfsc modifiying the expectation values,
    # the guaranteed result is returned for a noiseless simulator
    me = gen_DFSC_MitEx(AerBackend())

    c0 = (Circuit(3).X(0).X(1), 10, SymbolsDict())
    c1 = (Circuit(3).X(1).X(2), 10, SymbolsDict())
    qps_12 = QubitPauliString([Qubit(1), Qubit(2)], [Pauli.Z, Pauli.Z])
    qps_01 = QubitPauliString([Qubit(0), Qubit(1)], [Pauli.Z, Pauli.Z])
    qps_012 = QubitPauliString(
        [Qubit(0), Qubit(1), Qubit(2)], [Pauli.Z, Pauli.Z, Pauli.Z]
    )
    t0 = ObservableTracker(QubitPauliOperator({qps_01: 0.5, qps_12: 1.0}))
    t1 = ObservableTracker(QubitPauliOperator({qps_012: 0.7}))

    experiment = [ObservableExperiment(c0, t0), ObservableExperiment(c1, t1)]
    res = me.run(experiment)
    # correct expectations
    assert len(res) == 2
    assert res[0][qps_01] == 0.5
    assert res[0][qps_12] == -1.0
    assert res[1][qps_012] == 0.7


if __name__ == "__main__":
    test_get_clifford_mcs()
    test_prep_circuit_for_partition()
    test_DFSC_circuit_task_gen()
    test_DFSC_collater_task_gen()
    test_DFSC_characterisation_task_gen()
    test_DFSC_correction_task_gen()
    test_DFSC_mitex_gen()

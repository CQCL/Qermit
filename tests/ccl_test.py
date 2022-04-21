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

from pytket.predicates import CliffordCircuitPredicate  # type: ignore
from pytket import Circuit, OpType, Qubit
from qermit import (  # type: ignore
    ObservableTracker,
    SymbolsDict,
    MitEx,
    ObservableExperiment,
    AnsatzCircuit,
)
from qermit.clifford_noise_characterisation import (  # type: ignore
    gen_CDR_MitEx,
)
from qermit.clifford_noise_characterisation.ccl import (  # type: ignore
    sample_weighted_clifford_angle,
    gen_state_circuits,
    ccl_state_task_gen,
    ccl_result_batching_task_gen,
    ccl_likelihood_filtering_task_gen,
)
from pytket.extensions.qiskit import AerBackend  # type: ignore
from pytket.pauli import QubitPauliString, Pauli  # type: ignore
from pytket.utils import QubitPauliOperator
import numpy as np


def test_sample_weighted_clifford_angle():
    assert sample_weighted_clifford_angle(0.0, seed=10) == 0.0
    assert sample_weighted_clifford_angle(0.6, seed=10) == 0.5
    assert sample_weighted_clifford_angle(0.9, seed=10) == 1.0
    assert sample_weighted_clifford_angle(1.5, seed=10) == 1.5


def count_rzs(circuit):
    all_coms = circuit.get_commands()
    clifford_angles = set({0.5, 1.0, 1.5, 0, 2.0})
    non_cliff_counter = 0
    cliff_counter = 0
    for com in all_coms:
        if com.op.type == OpType.Rz:
            if com.op.params[0] not in clifford_angles:
                non_cliff_counter += 1
            else:
                cliff_counter += 1
    return (cliff_counter, non_cliff_counter)


def test_gen_state_circuits():
    # create dummy test circuit
    c = Circuit(4).Rz(0.9, 3).Rz(0.63, 1).Rx(0.2, 0).Rx(0.1, 2).Rz(3, 1)
    state_circuits0 = gen_state_circuits(
        c, n_non_cliffords=1, n_pairs=1, total_state_circuits=3, seed=197
    )
    assert len(state_circuits0) == 3
    # each state circuit should have
    s_0_coms = state_circuits0[0].get_commands()
    s_1_coms = state_circuits0[1].get_commands()
    s_2_coms = state_circuits0[2].get_commands()

    # manually assert angles of gates are expected for given state circuits for given seed
    # state circuit 0
    assert s_0_coms[5].op.params == [0.63]
    assert s_0_coms[7].op.params == [0.0]
    assert s_0_coms[8].op.params == [0.0]
    assert s_0_coms[10].op.params == [0.5]
    # state circuit 1
    assert s_1_coms[5].op.params == [2.0]
    assert s_1_coms[7].op.params == [0.0]
    assert s_1_coms[8].op.params == [0.0]
    assert s_1_coms[10].op.params == [0.1]
    # state circuit 2
    assert s_2_coms[5].op.params == [2.0]
    assert s_2_coms[7].op.params == [0.9]
    assert s_2_coms[8].op.params == [0.0]
    assert s_2_coms[10].op.params == [0.5]

    rz_counts_0 = count_rzs(state_circuits0[0])
    # 0th element number of rz cliffs, 1st element number of rz non cliffs
    assert rz_counts_0[0] == 3
    assert rz_counts_0[1] == 2

    state_circuits1 = gen_state_circuits(
        c, n_non_cliffords=2, n_pairs=2, total_state_circuits=2, seed=197
    )
    rz_counts_10 = count_rzs(state_circuits1[0])
    rz_counts_11 = count_rzs(state_circuits1[1])
    assert rz_counts_10[0] == 2
    assert rz_counts_10[1] == 3
    assert rz_counts_11[0] == 2
    assert rz_counts_11[1] == 3

    big_c = Circuit(10)
    for _ in range(10):
        big_c.Rz(0.9, 0).Rz(0.63, 1).Rx(0.2, 2).Rx(0.1, 3)
        big_c.Rz(0.9, 4).Rz(0.63, 5).Rx(0.2, 6).Rx(0.1, 7)
        big_c.Rz(0.9, 8).Rz(0.63, 9)
    num_commands = len(big_c.get_commands())
    num_non_cliffs = 10
    big_state_circuits = gen_state_circuits(
        big_c,
        n_non_cliffords=num_non_cliffs,
        n_pairs=8,
        total_state_circuits=10,
        seed=184,
    )

    for bc in big_state_circuits:
        assert count_rzs(bc) == (num_commands - num_non_cliffs, num_non_cliffs)

    big_state_circuits_cliff = gen_state_circuits(
        big_c,
        n_non_cliffords=0,
        n_pairs=0,
        total_state_circuits=10,
        seed=184,
    )

    for bc in big_state_circuits_cliff:
        assert CliffordCircuitPredicate().verify(bc)


def test_ccl_state_task_gen():
    num_non_cliffs = 2
    tot_state_circuits = 2

    task = ccl_state_task_gen(
        n_non_cliffords=num_non_cliffs,
        n_pairs=2,
        total_state_circuits=tot_state_circuits,
        simulator_backend=AerBackend(),
        tolerance=0.01,
        max_state_circuits_attempts=10,
    )
    assert task.n_in_wires == 1
    assert task.n_out_wires == 3

    c = Circuit(3).Rz(0.63, 1).Rz(0.2, 0).Rz(0.1, 2)
    c.Rz(0.43, 1).Rz(0.5, 0).Rz(0.8, 2)
    c.Rz(1.23, 1).Rz(1.2, 0).add_barrier([0, 1]).Rz(1.1, 2)
    c.Rz(9.63, 1).Rz(8.2, 0).Rz(10.1, 2)

    qps_012 = QubitPauliString(
        [Qubit(0), Qubit(1), Qubit(2)], [Pauli.Z, Pauli.Z, Pauli.Z]
    )
    t = ObservableTracker(QubitPauliOperator({qps_012: 1.0}))
    ac = AnsatzCircuit(c, 50, SymbolsDict())
    wire = [ObservableExperiment(ac, t)]

    res = task([wire])
    # first wire is the same experiment wire
    assert len(res) == 3
    assert res[0] == wire
    # second and third wire are the same, but to be sent through different MitEx objects
    # corresponding to noisy and noiselss backends
    assert len(res[2]) == tot_state_circuits
    assert len(res[1]) == tot_state_circuits
    # assert circuits are the same and ObservableTracker operators are the same
    assert res[1][0][0][0] == res[2][0][0][0]
    assert res[1][1][0][0] == res[2][1][0][0]

    assert count_rzs(res[1][0][0][0]) == (10, num_non_cliffs)
    assert count_rzs(res[1][1][0][0]) == (10, num_non_cliffs)

    assert res[1][0][1].qubit_pauli_operator == res[2][0][1].qubit_pauli_operator
    assert res[1][1][1].qubit_pauli_operator == res[2][1][1].qubit_pauli_operator


def test_result_batching_task_gen():
    # create dummy experiment with noiseless backend and single operator
    c = Circuit(3).Rz(0.63, 1).Rz(0.2, 0).Rz(0.1, 2)
    c.Rz(0.43, 1).Rz(0.5, 0).Rz(0.8, 2)
    c.Rz(1.23, 1).Rz(1.2, 0).Rz(1.1, 2)
    c.Rz(9.63, 1).Rz(8.2, 0).Rz(10.1, 2)

    qps_012 = QubitPauliString(
        [Qubit(0), Qubit(1), Qubit(2)], [Pauli.Z, Pauli.Z, Pauli.Z]
    )
    qps_01 = QubitPauliString([Qubit(0), Qubit(1)], [Pauli.Z, Pauli.Z])
    t0 = ObservableTracker(QubitPauliOperator({qps_012: 1.0}))
    ac0 = AnsatzCircuit(c, 10, SymbolsDict())

    t1 = ObservableTracker(QubitPauliOperator({qps_01: 0.5}))
    ac1 = AnsatzCircuit(c.copy(), 10, SymbolsDict())

    b = AerBackend()

    n_state_circuits = 10
    res = ccl_state_task_gen(
        n_non_cliffords=2,
        n_pairs=2,
        total_state_circuits=n_state_circuits,
        simulator_backend=b,
        tolerance=0.01,
        max_state_circuits_attempts=10,
    )([[ObservableExperiment(ac0, t0), ObservableExperiment(ac1, t1)]])
    mitex = MitEx(backend=b)
    qpos_noiseless = mitex.run(res[1])
    qpos_noisy = mitex.run(res[2])

    task = ccl_result_batching_task_gen(n_state_circuits)
    assert task.n_in_wires == 2
    assert task.n_out_wires == 1

    wire = (qpos_noiseless, qpos_noisy)
    batching_res = task(wire)[0]

    # each original experiment should have a different set of characteriastion results
    assert len(batching_res) == 2

    # for each state circuit there should then be some pair of qubit pauli operators
    assert len(batching_res[0]) == n_state_circuits
    assert len(batching_res[1]) == n_state_circuits

    # and each entry should just be a pair of qubit pauli operators
    assert len(batching_res[0][0]) == 2
    assert batching_res[0][0][0] == QubitPauliOperator({qps_012: 1.0})
    assert len(batching_res[1][0]) == 2
    assert batching_res[1][0][0] == QubitPauliOperator({qps_01: 0.5})


def test_ccl_likelihood_filtering_task_gen():
    def likelihood_function(arg0, arg1) -> float:
        # set up function such that for one experiment it returns all operators, and
        # for the other experiment it returns no operators
        if float(sum(arg0._dict.values())) > 0.6:
            return 0
        else:
            return 1

    # set up dummy experiments
    qps_012 = QubitPauliString(
        [Qubit(0), Qubit(1), Qubit(2)], [Pauli.Z, Pauli.Z, Pauli.Z]
    )
    qps_01 = QubitPauliString([Qubit(0), Qubit(1)], [Pauli.Z, Pauli.Z])

    qpo_012 = QubitPauliOperator({qps_012: 0.5})
    qpo_01 = QubitPauliOperator({qps_01: 1.0})

    assert likelihood_function(qpo_012, 0) == 1.0
    assert likelihood_function(qpo_01, 100) == 0.0

    task = ccl_likelihood_filtering_task_gen(
        likelihood_function=likelihood_function, seed=10
    )
    assert task.n_in_wires == 1
    assert task.n_out_wires == 1
    wire = [
        [(qpo_012, qpo_012), (qpo_01, qpo_01)],
        [(qpo_01, qpo_01), (qpo_012, qpo_012)],
    ]
    filtered_res = task([wire])[0]
    assert filtered_res == [[(qpo_012, qpo_012)], [(qpo_012, qpo_012)]]


def test_cdr_mitex():
    c = Circuit(3).Rz(0.63, 1).Rz(0.2, 0).Rz(0.1, 2)
    c.Rz(0.43, 1).Rz(0.5, 0).Rz(0.8, 2)
    c.Rz(1.23, 1).Rz(1.2, 0).Rz(1.1, 2)
    c.Rz(9.63, 1).Rz(8.2, 0).Rz(10.1, 2)

    qps_012 = QubitPauliString(
        [Qubit(0), Qubit(1), Qubit(2)], [Pauli.Z, Pauli.Z, Pauli.Z]
    )
    qps_01 = QubitPauliString([Qubit(0), Qubit(1)], [Pauli.Z, Pauli.Z])
    t0 = ObservableTracker(QubitPauliOperator({qps_012: 1.0}))
    ac0 = AnsatzCircuit(c, 10, SymbolsDict())

    t1 = ObservableTracker(QubitPauliOperator({qps_01: 0.5}))
    ac1 = AnsatzCircuit(c.copy(), 10, SymbolsDict())

    b = AerBackend()
    b._characterisation = dict()
    ccl_me = gen_CDR_MitEx(b, b, n_non_cliffords=2, n_pairs=2, total_state_circuits=20)
    ccl_res = ccl_me.run([ObservableExperiment(ac0, t0), ObservableExperiment(ac1, t1)])

    me = MitEx(b)
    res = me.run([(ac0, t0), (ac1, t1)])
    # check that mitigated and non mitigated versions return identical values
    # for noiseless simulation with same experiment
    assert len(ccl_res) == 2
    assert len(res) == 2

    assert np.isclose(float(ccl_res[0][qps_012]), float(res[0][qps_012]), rtol=1e-1)
    assert np.isclose(float(ccl_res[1][qps_01]), float(res[1][qps_01]), rtol=1e-1)


if __name__ == "__main__":
    test_sample_weighted_clifford_angle()
    test_gen_state_circuits()
    test_ccl_state_task_gen()
    test_result_batching_task_gen()
    test_ccl_likelihood_filtering_task_gen()
    test_cdr_mitex()

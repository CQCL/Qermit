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

from qermit import (  # type: ignore
    ObservableTracker,
    SymbolsDict,
)
from qermit.zero_noise_extrapolation import (  # type: ignore
    Folding,
    Fit,
    gen_ZNE_MitEx,
)
from qermit.zero_noise_extrapolation.zne import (  # type: ignore
    gen_initial_compilation_task,
    gen_duplication_task,
    extrapolation_task_gen,
    digital_folding_task_gen,
    gen_qubit_relabel_task,
    merge_experiments_task_gen,
)
from pytket.predicates import GateSetPredicate  # type: ignore
from pytket.extensions.qiskit import AerBackend, IBMQEmulatorBackend  # type: ignore
from pytket import Circuit, Qubit
from pytket.pauli import Pauli, QubitPauliString  # type: ignore
from pytket.utils import QubitPauliOperator
from pytket.circuit import CircBox
from numpy.polynomial.polynomial import polyval  # type: ignore
import math
import numpy as np  # type: ignore
from qermit import AnsatzCircuit, ObservableExperiment  # type: ignore
import qiskit_aer.noise as noise  # type: ignore
from pytket.circuit import OpType  # type: ignore
from qiskit_ibm_provider import IBMProvider
import pytest
from pytket.circuit import Node  # type: ignore
from qermit.noise_model import MockQuantinuumBackend  # type: ignore
from qermit.taskgraph import gen_MeasurementReduction_MitEx
from qermit.noise_model import NoiseModel, ErrorDistribution
from qermit.zero_noise_extrapolation.zne import gen_noise_scaled_mitex
from qermit.noise_model import TranspilerBackend, PauliErrorTranspile
from itertools import product
import multiprocessing as mp

n_qubits = 2

prob_1 = 0.005
prob_2 = 0.02

noise_model = noise.NoiseModel()

# Depolarizing quantum errors
error_2 = noise.depolarizing_error(prob_2, 2)
for edge in [[i, j] for i in range(n_qubits) for j in range(n_qubits)]:
    if edge[0] != edge[1]:
        noise_model.add_quantum_error(error_2, ["cx"], [edge[0], edge[1]])

error_1 = noise.depolarizing_error(prob_1, 1)
for node in [i for i in range(n_qubits)]:
    noise_model.add_quantum_error(error_1, ["h", "rx", "u3"], [node])

noisy_backend = AerBackend(noise_model)


skip_remote_tests: bool = not IBMProvider.saved_accounts()
REASON = "IBMQ account not configured"


def get_string_operator(meas_qubits, string):
    # A small utility method for generating observables with expectation
    # corresponding to bit string probabilities.

    identity_pauli_list = [Pauli.I for _ in meas_qubits]

    identity_qps = QubitPauliString(
        qubits=meas_qubits,
        paulis=identity_pauli_list,
    )
    qpo = QubitPauliOperator({identity_qps: 1})

    for i in range(len(meas_qubits)):
        temp_pauli_list = identity_pauli_list.copy()
        temp_pauli_list[i] = Pauli.Z
        temp_qps = QubitPauliString(
            qubits=meas_qubits,
            paulis=temp_pauli_list,
        )
        if string[i] == 0:
            qpo *= QubitPauliOperator({temp_qps: 0.5, identity_qps: 0.5})
        elif string[i] == 1:
            qpo *= QubitPauliOperator({temp_qps: -0.5, identity_qps: 0.5})
        else:
            raise Exception(f"{string} is not a binary string.")

    return qpo


@pytest.mark.high_compute
def test_measurement_reduction_integration():

    tol = 0.01

    # A bell state
    circuit = Circuit()
    qubit_0 = Qubit(name='my_qubit', index=0)
    qubit_1 = Qubit(name='my_qubit', index=1)
    circuit.add_qubit(qubit_0)
    circuit.add_qubit(qubit_1)
    circuit.H(qubit_0).CX(qubit_0, qubit_1)
    meas_qubits = [qubit_0, qubit_1]

    backend = AerBackend()
    reduction_mitex = gen_MeasurementReduction_MitEx(backend=backend)

    ansatz = AnsatzCircuit(
        Circuit=circuit,
        Shots=1000000,
        SymbolsDict=SymbolsDict()
    )

    qpo = get_string_operator(meas_qubits, [0, 0])
    obs = ObservableTracker(qubit_pauli_operator=qpo)
    obs_exp = ObservableExperiment(AnsatzCircuit=ansatz, ObservableTracker=obs)
    result = reduction_mitex.run(
        mitex_wires=[obs_exp]
    )
    # The probability of measuring 00 is 0.5
    assert abs(sum(result[0]._dict.values()) - 0.5) < tol

    qpo = get_string_operator(meas_qubits, [0, 1])
    obs = ObservableTracker(qubit_pauli_operator=qpo)
    obs_exp = ObservableExperiment(AnsatzCircuit=ansatz, ObservableTracker=obs)
    result = reduction_mitex.run(
        mitex_wires=[obs_exp]
    )
    # The probability of measuring 01 is 0
    assert abs(sum(result[0]._dict.values())) < tol

    folding_type = Folding.two_qubit_gate
    fit_type = Fit.linear
    noise_scaling_list = [1, 1.5, 2, 2.5, 3, 3.5]
    zne_mitex = gen_ZNE_MitEx(
        backend=backend,
        experiment_mitex=reduction_mitex,
        noise_scaling_list=noise_scaling_list,
        folding_type=folding_type,
        fit_type=fit_type,
    )

    qpo = get_string_operator(meas_qubits, [1, 0])
    obs = ObservableTracker(qubit_pauli_operator=qpo)
    obs_exp = ObservableExperiment(AnsatzCircuit=ansatz, ObservableTracker=obs)
    result = zne_mitex.run(
        mitex_wires=[obs_exp]
    )
    # The probability of measuring 10 is 0
    assert abs(sum(result[0]._dict.values())) < tol

    qpo = get_string_operator(meas_qubits, [1, 1])
    obs = ObservableTracker(qubit_pauli_operator=qpo)
    obs_exp = ObservableExperiment(AnsatzCircuit=ansatz, ObservableTracker=obs)
    result = zne_mitex.run(
        mitex_wires=[obs_exp]
    )
    # The probability of measuring 11 is 0.5
    assert abs(sum(result[0]._dict.values()) - 0.5) < tol


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.high_compute
def test_no_qubit_relabel():

    lagos_backend = IBMQEmulatorBackend("ibmq_mumbai")
    zne_mitex = gen_ZNE_MitEx(backend=lagos_backend, noise_scaling_list=[3, 5, 7])

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
    result = zne_mitex.run(exp)[0]
    assert result.all_qubits == {Qubit(0), Qubit(1), Qubit(2)}


def test_gen_qubit_relabel_task():

    task = gen_qubit_relabel_task()

    assert task.n_in_wires == 2
    assert task.n_out_wires == 1

    qubit_pauli_string = QubitPauliString(
        [Qubit(0), Qubit(1), Qubit(2)], [Pauli.Z, Pauli.Z, Pauli.Z]
    )
    qubit_pauli_operator = QubitPauliOperator({qubit_pauli_string: 1.0})

    compilation_map = [{Node(0): Qubit(0), Node(1): Qubit(1), Node(2): Qubit(2)}]

    relabeled_qubit_pauli_string = QubitPauliString(
        [Node(0), Node(1), Node(2)], [Pauli.Z, Pauli.Z, Pauli.Z]
    )
    relabeled_qubit_pauli_operator = QubitPauliOperator(
        {relabeled_qubit_pauli_string: 1.0}
    )

    result = task(([qubit_pauli_operator], compilation_map))[0][0]
    assert result == relabeled_qubit_pauli_operator


def test_gen_initial_compilation_task():

    be = AerBackend()

    task = gen_initial_compilation_task(be, optimisation_level=1)

    assert task.n_in_wires == 1
    assert task.n_out_wires == 2

    c_1 = Circuit(2).CZ(0, 1).T(1)
    c_2 = Circuit(1).T(0).X(0)

    ac_1 = AnsatzCircuit(c_1, 10000, {})
    ac_2 = AnsatzCircuit(c_2, 10000, {})

    qpo_1 = QubitPauliOperator({QubitPauliString([Qubit(1)], [Pauli.Z]): 1})
    qpo_2 = QubitPauliOperator({QubitPauliString([Qubit(0)], [Pauli.Z]): 1})

    experiment_1 = ObservableExperiment(ac_1, ObservableTracker(qpo_1))
    experiment_2 = ObservableExperiment(ac_2, ObservableTracker(qpo_2))

    result = task([[experiment_1, experiment_2]])

    compiled_experiment_1 = result[0][0]
    compiled_experiment_2 = result[0][1]

    compiled_c_1 = compiled_experiment_1[0][0]
    compiled_c_2 = compiled_experiment_2[0][0]

    # Check that the compiled circuits are indeed valid
    assert be.valid_circuit(compiled_c_1)
    assert be.valid_circuit(compiled_c_2)


def test_gen_initial_compilation_task_quantinuum_qubit_names():
    be = MockQuantinuumBackend()

    task = gen_initial_compilation_task(be, optimisation_level=0)

    assert task.n_in_wires == 1
    assert task.n_out_wires == 2

    c = Circuit(2).X(0).X(1)

    ac = AnsatzCircuit(c, 10000, {})

    qpo = QubitPauliOperator({QubitPauliString([Qubit(0)], [Pauli.Z]): 1})

    experiment = ObservableExperiment(ac, ObservableTracker(qpo))

    result = task([[experiment]])

    c_qubits = set(result[0][0].AnsatzCircuit.Circuit.qubits)
    qpo_qubits = result[0][0].ObservableTracker._qubit_pauli_operator.all_qubits
    assert qpo_qubits.issubset(c_qubits)


def test_gen_duplication_task():

    n_dups = 2

    task = gen_duplication_task(n_dups)

    assert task.n_in_wires == 1
    assert task.n_out_wires == n_dups

    c_1 = Circuit(2).CZ(0, 1).T(1)
    c_2 = Circuit(2).CZ(0, 1).T(0).X(1)

    ac_1 = AnsatzCircuit(c_1, 10000, {})
    ac_2 = AnsatzCircuit(c_2, 10000, {})

    qpo_1 = QubitPauliOperator({QubitPauliString([Qubit(0)], [Pauli.Z]): 1})
    qpo_2 = QubitPauliOperator({QubitPauliString([Qubit(1)], [Pauli.Z]): 1})

    experiment_1 = ObservableExperiment(ac_1, ObservableTracker(qpo_1))
    experiment_2 = ObservableExperiment(ac_2, ObservableTracker(qpo_2))

    result = task([[experiment_1, experiment_2]])

    duplicate_1 = result[0]
    duplicate_2 = result[1]

    for duplicate_1_experiment, duplicate_2_experiment in zip(duplicate_1, duplicate_2):

        duplicate_1_ac = duplicate_1_experiment[0]
        duplicate_1_qpo = duplicate_1_experiment[1]

        duplicate_2_ac = duplicate_2_experiment[0]
        duplicate_2_qpo = duplicate_2_experiment[1]

        assert duplicate_1_ac == duplicate_2_ac
        assert (
            duplicate_1_qpo._qubit_pauli_operator
            == duplicate_2_qpo._qubit_pauli_operator
        )


def test_extrapolation_task_gen():

    n_folds = [1, 2, 3, 4, 5]

    task = extrapolation_task_gen(n_folds, Fit.polynomial, False, 2)

    assert task.n_in_wires == len(n_folds)
    assert task.n_out_wires == 1

    # Defines the function (x/10 - 1)**2
    coef_1 = [1, -2 / 10, 1 / 100]
    # Defines the function -(x/10 - 1)**2
    coef_2 = [-1, 2 / 10, -1 / 100]

    # These function return pauli operators with expectation values depending on noise levels
    def qpo_1(noise_level):
        return QubitPauliOperator(
            {QubitPauliString([Qubit(0)], [Pauli.Z]): polyval(noise_level, coef_1)}
        )

    def qpo_2(noise_level):
        return QubitPauliOperator(
            {QubitPauliString([Qubit(1)], [Pauli.X]): polyval(noise_level, coef_2)}
        )

    # Expectation values at defined noise scaling
    args = [[qpo_1(i), qpo_2(i)] for i in n_folds]
    args = tuple(args)

    result = task(args)[0]

    experiment_1_result = result[0]._dict
    experiment_2_result = result[1]._dict

    # Check that the expectation values are as they would be in the ideal case.
    # The coefficients defined above intersect at 1 and -1 which we take to be the ideal values.
    assert math.isclose(
        experiment_1_result[list(experiment_1_result.keys())[0]], 1, rel_tol=0.001
    )
    assert math.isclose(
        experiment_2_result[list(experiment_2_result.keys())[0]], -1, rel_tol=0.001
    )


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.high_compute
def test_folding_compiled_circuit():

    emulator_backend = IBMQEmulatorBackend("ibmq_mumbai")

    n_folds_1 = 3

    task_1 = digital_folding_task_gen(
        emulator_backend,
        n_folds_1,
        Folding.circuit,
        _allow_approx_fold=False,
    )

    assert task_1.n_in_wires == 1
    assert task_1.n_out_wires == 2

    c_1 = Circuit(1).Rz(3.5, 0)
    c_1 = emulator_backend.get_compiled_circuit(c_1)

    ac_1 = AnsatzCircuit(c_1, 10000, {})

    qpo_1 = QubitPauliOperator({QubitPauliString([Qubit(0)], [Pauli.Z]): 1})

    experiment_1 = ObservableExperiment(ac_1, ObservableTracker(qpo_1))

    folded_experiment_1 = task_1([[experiment_1]])[0][0]
    assert OpType.Reset not in [
        com.op.type for com in folded_experiment_1.AnsatzCircuit.Circuit.get_commands()
    ]


def test_digital_folding_task_gen():

    be = AerBackend()

    n_folds_1 = 5
    n_folds_2 = 3
    n_folds_3 = 6
    n_folds_4 = 2

    task_1 = digital_folding_task_gen(
        be, n_folds_1, Folding.circuit, _allow_approx_fold=False
    )
    task_2 = digital_folding_task_gen(
        be, n_folds_2, Folding.gate, _allow_approx_fold=False
    )
    task_3 = digital_folding_task_gen(
        noisy_backend,
        n_folds_3,
        Folding.gate,
        _allow_approx_fold=False,
    )
    task_4 = digital_folding_task_gen(
        noisy_backend,
        n_folds_4,
        Folding.odd_gate,
        _allow_approx_fold=False,
    )

    assert task_1.n_in_wires == 1
    assert task_1.n_out_wires == 2
    assert task_2.n_in_wires == 1
    assert task_2.n_out_wires == 2
    assert task_3.n_in_wires == 1
    assert task_3.n_out_wires == 2
    assert task_4.n_in_wires == 1
    assert task_4.n_out_wires == 2

    c_1 = Circuit(2).CZ(0, 1).T(1)
    c_2 = Circuit(2).CZ(0, 1).T(0).X(1)
    c_3 = Circuit(2).CX(0, 1).H(0).Rx(0.3, 1).Rz(0.6, 1)
    c_4 = Circuit(2).CX(0, 1).H(0).Rz(0.3, 1)
    c_5 = Circuit(2).H(0).add_barrier([0, 1]).CX(0, 1)

    ac_1 = AnsatzCircuit(c_1, 10000, {})
    ac_2 = AnsatzCircuit(c_2, 10000, {})
    ac_3 = AnsatzCircuit(c_3, 10000, {})
    ac_4 = AnsatzCircuit(c_4, 10000, {})
    ac_5 = AnsatzCircuit(c_5, 10000, {})

    qpo_1 = QubitPauliOperator({QubitPauliString([Qubit(0)], [Pauli.Z]): 1})
    qpo_2 = QubitPauliOperator({QubitPauliString([Qubit(1)], [Pauli.Z]): 1})
    qpo_3 = QubitPauliOperator(
        {QubitPauliString([Qubit(0), Qubit(1)], [Pauli.Z, Pauli.Z]): 1}
    )
    qpo_4 = QubitPauliOperator({QubitPauliString([Qubit(1)], [Pauli.Z]): 1})

    experiment_1 = ObservableExperiment(ac_1, ObservableTracker(qpo_1))
    experiment_2 = ObservableExperiment(ac_2, ObservableTracker(qpo_2))
    experiment_3 = ObservableExperiment(ac_3, ObservableTracker(qpo_3))
    experiment_4 = ObservableExperiment(ac_4, ObservableTracker(qpo_4))
    experiment_5 = ObservableExperiment(ac_5, ObservableTracker(qpo_3))

    folded_experiment_1 = task_1([[experiment_1]])[0][0]
    folded_experiment_2 = task_2([[experiment_2]])[0][0]
    folded_experiment_3 = task_3([[experiment_3]])[0][0]
    folded_experiment_4 = task_4([[experiment_4]])[0][0]
    folded_experiment_5 = task_1([[experiment_5]])[0][0]
    folded_experiment_6 = task_2([[experiment_5]])[0][0]
    folded_experiment_7 = task_4([[experiment_5]])[0][0]

    folded_c_1 = folded_experiment_1[0][0]
    folded_c_2 = folded_experiment_2[0][0]
    folded_c_3 = folded_experiment_3[0][0]
    folded_c_4 = folded_experiment_4[0][0]
    folded_c_5 = folded_experiment_5[0][0]
    folded_c_6 = folded_experiment_6[0][0]
    folded_c_7 = folded_experiment_7[0][0]

    # TODO: Add a backend with a more restricted gateset
    assert GateSetPredicate(be.backend_info.gate_set).verify(folded_c_1)
    assert GateSetPredicate(be.backend_info.gate_set).verify(folded_c_2)
    assert GateSetPredicate(noisy_backend.backend_info.gate_set).verify(folded_c_3)
    assert GateSetPredicate(noisy_backend.backend_info.gate_set).verify(folded_c_4)
    assert GateSetPredicate(be.backend_info.gate_set).verify(folded_c_5)
    assert GateSetPredicate(be.backend_info.gate_set).verify(folded_c_6)
    assert GateSetPredicate(noisy_backend.backend_info.gate_set).verify(folded_c_7)

    # Checks that the number of gates has been increased correctly.
    # Note that in both cases barriers are added. This is why there is the
    # n_folds_i - 1 term at the end.
    assert folded_c_1.n_gates == c_1.n_gates * n_folds_1 + n_folds_1 - 1
    assert folded_c_2.n_gates == c_2.n_gates * n_folds_2 + c_2.n_gates * (n_folds_2 - 1)
    assert folded_c_3.n_gates == c_3.n_gates * n_folds_3 + c_3.n_gates * (n_folds_3 - 1)
    assert folded_c_4.n_gates == c_4.n_gates + n_folds_4 * 2 * ((c_4.n_gates + 1) // 2)
    assert folded_c_5.n_gates == c_5.n_gates * n_folds_1 + n_folds_1 - 1
    assert folded_c_6.n_gates == (
        c_5.n_gates - c_5.n_gates_of_type(OpType.Barrier)
    ) * n_folds_2 + (c_5.n_gates - c_5.n_gates_of_type(OpType.Barrier)) * (
        n_folds_2 - 1
    ) + c_5.n_gates_of_type(
        OpType.Barrier
    )
    assert folded_c_7.n_gates == c_5.n_gates + n_folds_4 * 2 * (
        ((c_5.n_gates - c_5.n_gates_of_type(OpType.Barrier)) + 1) // 2
    )

    c_1_unitary = c_1.get_unitary()
    c_2_unitary = c_2.get_unitary()
    c_3_unitary = c_3.get_unitary()
    c_4_unitary = c_4.get_unitary()
    c_5_unitary = c_5.get_unitary()
    folded_c_1_unitary = folded_c_1.get_unitary()
    folded_c_2_unitary = folded_c_2.get_unitary()
    folded_c_3_unitary = folded_c_3.get_unitary()
    folded_c_4_unitary = folded_c_4.get_unitary()
    folded_c_5_unitary = folded_c_5.get_unitary()
    folded_c_6_unitary = folded_c_6.get_unitary()
    folded_c_7_unitary = folded_c_7.get_unitary()

    assert np.allclose(c_1_unitary, folded_c_1_unitary)
    assert np.allclose(c_2_unitary, folded_c_2_unitary)
    assert np.allclose(c_3_unitary, folded_c_3_unitary)
    assert np.allclose(c_4_unitary, folded_c_4_unitary)
    assert np.allclose(c_5_unitary, folded_c_5_unitary)
    assert np.allclose(c_5_unitary, folded_c_6_unitary)
    assert np.allclose(c_5_unitary, folded_c_7_unitary)


def test_zne_identity():

    backend = AerBackend()

    me = gen_ZNE_MitEx(
        backend,
        [1, 7, 5, 3],
        _label="TestZNEMitEx",
        optimisation_level=0,
    )

    c = Circuit(3)
    for _ in range(10):
        c.X(0).X(1).X(2)
    ac = AnsatzCircuit(c, 100, SymbolsDict())

    qps = QubitPauliString([Qubit(0), Qubit(1), Qubit(2)], [Pauli.Z, Pauli.Z, Pauli.Z])

    qpo = QubitPauliOperator({qps: 1.0})

    x = me.run([ObservableExperiment(ac, ObservableTracker(qpo))])

    assert round(x[0]._dict[qps]) == 1


def test_simple_run_end_to_end():

    be = AerBackend()

    me = gen_ZNE_MitEx(
        be,
        [1, 2, 3, 4],
        _label="TestZNEMitEx",
        optimisation_level=0,
        folding_type=Folding.gate,
        show_fit=False,
    )

    c_1 = Circuit(2).CZ(0, 1).T(1)
    c_2 = Circuit(1).T(0).X(0)
    ac_1 = AnsatzCircuit(c_1, 10000, SymbolsDict())
    ac_2 = AnsatzCircuit(c_2, 10000, SymbolsDict())
    circ_list = []
    circ_list.append(
        ObservableExperiment(
            ac_1,
            ObservableTracker(
                QubitPauliOperator({QubitPauliString([Qubit(1)], [Pauli.Z]): 1})
            ),
        )
    )
    circ_list.append(
        ObservableExperiment(
            ac_2,
            ObservableTracker(
                QubitPauliOperator({QubitPauliString([Qubit(0)], [Pauli.Z]): 1})
            ),
        )
    )

    result = me.run(circ_list)
    expectation_1 = result[0]
    expectation_2 = result[1]

    res1 = expectation_1[QubitPauliString([Qubit(1)], [Pauli.Z])]
    res2 = expectation_2[QubitPauliString([Qubit(0)], [Pauli.Z])]

    assert round(float(res1)) == 1.0
    assert round(float(res2)) == -1.0


def test_circuit_folding_TK1():

    circ = Circuit(2)
    circ.add_gate(OpType.TK1, (0, 0.1, 0), [0])
    circ.CX(0, 1)

    folded_circ = Folding.circuit(circ, 3)[0]

    circ_unitary = circ.get_unitary()
    folded_circ_unitary = folded_circ.get_unitary()
    assert np.allclose(circ_unitary, folded_circ_unitary)


def test_odd_gate_folding():

    circ = Circuit(2).CX(0, 1).X(0).CX(1, 0).X(1)
    folded_circ = Folding.odd_gate(circ, 2)[0]
    correct_folded_circ = (
        Circuit(2)
        .CX(0, 1)
        .add_barrier([0, 1])
        .CX(0, 1)
        .add_barrier([0, 1])
        .CX(0, 1)
        .X(0)
        .CX(1, 0)
        .add_barrier([1, 0])
        .CX(1, 0)
        .add_barrier([1, 0])
        .CX(1, 0)
        .X(1)
    )
    assert folded_circ == correct_folded_circ

    circ = Circuit(3).CX(0, 1).CX(1, 2)
    folded_circ = Folding.odd_gate(circ, 3)[0]
    correct_folded_circ = (
        Circuit(3)
        .CX(0, 1)
        .add_barrier([0, 1])
        .CX(0, 1)
        .add_barrier([0, 1])
        .CX(0, 1)
        .add_barrier([0, 1])
        .CX(0, 1)
        .add_barrier([0, 1])
        .CX(0, 1)
        .CX(1, 2)
    )
    assert folded_circ == correct_folded_circ


def test_two_qubit_gate_folding():

    be = AerBackend()

    n_folds_1 = 3

    # This tests the case of odd integer folding, which is always possible
    task_1 = digital_folding_task_gen(
        be, n_folds_1, Folding.two_qubit_gate, _allow_approx_fold=False
    )
    # This tests the case of non integer folding. 5.5 is not possible
    # as the circuit has 1 gate and so only odd integer folding is possible
    task_invalid = digital_folding_task_gen(
        be, 5.5, Folding.two_qubit_gate, _allow_approx_fold=False
    )
    # This test the case of non integer folding when approximate folding is
    # allowed
    task_2 = digital_folding_task_gen(
        be, 5.8, Folding.two_qubit_gate, _allow_approx_fold=True
    )

    assert task_1.n_in_wires == 1
    assert task_1.n_out_wires == 2
    assert task_2.n_in_wires == 1
    assert task_2.n_out_wires == 2
    assert task_invalid.n_in_wires == 1
    assert task_invalid.n_out_wires == 2

    c_1 = Circuit(2).Rz(0.3, 0).ZZPhase(0.3, 1, 0)
    # This is to ensure that the barrier is not folded, even though it
    # acts on 2 qubits
    c_2 = Circuit(3).Rz(0.3, 2).CZ(1, 2).add_barrier([0, 1]).CX(0, 1).X(0)
    c_3 = Circuit(3).CZ(0, 1).CZ(1, 2).CZ(0, 2)

    circ_box = CircBox(c_1)
    # Tests that circuits with nothing to fold will raise an error.
    c_gate_set_invalid_1 = Circuit(2).add_circbox(circ_box, [0, 1])
    # Tests that circuits with CircBox will raise an error.
    c_gate_set_invalid_2 = Circuit(2).add_circbox(circ_box, [0, 1]).CZ(0, 1)

    ac_1 = AnsatzCircuit(c_1, 10000, {})
    ac_2 = AnsatzCircuit(c_2, 10000, {})
    ac_3 = AnsatzCircuit(c_3, 10000, {})
    ac_gate_set_invalid_1 = AnsatzCircuit(c_gate_set_invalid_1, 10000, {})
    ac_gate_set_invalid_2 = AnsatzCircuit(c_gate_set_invalid_2, 10000, {})

    qpo_1 = QubitPauliOperator({QubitPauliString([Qubit(0)], [Pauli.Z]): 1})

    experiment_1 = ObservableExperiment(ac_1, ObservableTracker(qpo_1))
    experiment_2 = ObservableExperiment(ac_2, ObservableTracker(qpo_1))
    experiment_3 = ObservableExperiment(ac_3, ObservableTracker(qpo_1))
    experiment_gate_set_invalid_1 = ObservableExperiment(
        ac_gate_set_invalid_1, ObservableTracker(qpo_1)
    )
    experiment_gate_set_invalid_2 = ObservableExperiment(
        ac_gate_set_invalid_2, ObservableTracker(qpo_1)
    )

    with pytest.raises(ValueError):
        task_invalid([[experiment_1, experiment_2]])

    with pytest.raises(RuntimeError):
        task_1([[experiment_gate_set_invalid_1]])

    with pytest.raises(RuntimeError):
        task_1([[experiment_gate_set_invalid_2]])

    folded_experiment_1 = task_1([[experiment_1, experiment_2]])[0]
    folded_experiment_2 = task_2([[experiment_3]])[0]

    folded_c_1 = folded_experiment_1[0].AnsatzCircuit.Circuit
    folded_c_2 = folded_experiment_1[1].AnsatzCircuit.Circuit
    folded_c_3 = folded_experiment_2[0].AnsatzCircuit.Circuit

    ideal_folded_c_1 = Circuit(2)
    ideal_folded_c_1.Rz(0.3, 0)
    ideal_folded_c_1.ZZPhase(0.3, 1, 0)
    ideal_folded_c_1.add_barrier([1, 0])
    ideal_folded_c_1.ZZPhase(3.7, 1, 0)
    ideal_folded_c_1.add_barrier([1, 0])
    ideal_folded_c_1.ZZPhase(0.3, 1, 0)

    assert folded_c_1 == ideal_folded_c_1

    assert GateSetPredicate(be.backend_info.gate_set).verify(folded_c_1)
    assert GateSetPredicate(be.backend_info.gate_set).verify(folded_c_2)
    assert GateSetPredicate(be.backend_info.gate_set).verify(folded_c_3)

    # Note that in both cases barriers are added. This is why there is the
    # n_folds_i - 1 term at the end.
    assert folded_c_1.n_gates == c_1.n_gates + 2 * c_1.n_2qb_gates() * (n_folds_1 - 1)
    assert folded_c_2.n_gates == c_2.n_gates + 2 * c_2.n_2qb_gates() * (n_folds_1 - 1)
    # note that this gives an nose scaling = 17/3 = 5.6 which is a little smaller than 5.8
    assert folded_c_3.n_2qb_gates() == 17

    c_1_unitary = c_1.get_unitary()
    c_2_unitary = c_2.get_unitary()
    c_3_unitary = c_3.get_unitary()
    folded_c_1_unitary = folded_c_1.get_unitary()
    folded_c_2_unitary = folded_c_2.get_unitary()
    folded_c_3_unitary = folded_c_3.get_unitary()

    assert np.allclose(c_1_unitary, folded_c_1_unitary)
    assert np.allclose(c_2_unitary, folded_c_2_unitary)
    assert np.allclose(c_3_unitary, folded_c_3_unitary)


@pytest.mark.high_compute
def test_end_to_end_noise_scaled_mitex():

    error_rate = 0.1
    error_distribution = ErrorDistribution(
        distribution={
            (Pauli.X, Pauli.I): error_rate,
        }
    )
    noise_model = NoiseModel(
        noise_model={
            OpType.CZ: error_distribution
        }
    )

    noise_scaling = 5
    noise_scaled_mitex = gen_noise_scaled_mitex(
        backend=AerBackend(),
        noise_scaling=noise_scaling,
        folding_type=Folding.noise_aware,
        noise_model=noise_model,
        n_noisy_circuit_samples=10000,
    )

    circuit_noisy = Circuit(2).CZ(0, 1)
    circuit_noisless = Circuit(2).CX(0, 1)

    qps_noisy_noisy = QubitPauliString(map={Qubit(0): Pauli.Z, Qubit(1): Pauli.Z})
    qps_noisy_noisless = QubitPauliString(map={Qubit(0): Pauli.X})

    qubit_pauli_operator_noisy = QubitPauliOperator(
        dictionary={
            qps_noisy_noisy: 1,
            qps_noisy_noisless: 1,
        }
    )

    qps_noisless_one = QubitPauliString(map={Qubit(0): Pauli.Z})
    qps_noisless_zero = QubitPauliString(map={Qubit(1): Pauli.X})

    qubit_pauli_operator_noisless = QubitPauliOperator(
        dictionary={
            qps_noisless_one: 1,
            qps_noisless_zero: 1,
        }
    )

    observable_tracker_noisy = ObservableTracker(
        qubit_pauli_operator=qubit_pauli_operator_noisy
    )
    observable_tracker_noisless = ObservableTracker(
        qubit_pauli_operator=qubit_pauli_operator_noisless
    )

    shots = 1000000
    ansatz_circuit_noisy = AnsatzCircuit(
        Circuit=circuit_noisy,
        Shots=shots,
        SymbolsDict={}
    )
    ansatz_circuit_noisless = AnsatzCircuit(
        Circuit=circuit_noisless,
        Shots=shots,
        SymbolsDict={}
    )

    observable_experiment_noisy = ObservableExperiment(
        AnsatzCircuit=ansatz_circuit_noisy,
        ObservableTracker=observable_tracker_noisy,
    )
    observable_experiment_noisless = ObservableExperiment(
        AnsatzCircuit=ansatz_circuit_noisless,
        ObservableTracker=observable_tracker_noisless,
    )

    qubit_pauli_operator_list = noise_scaled_mitex.run(
        mitex_wires=[observable_experiment_noisy, observable_experiment_noisless]
    )
    assert qubit_pauli_operator_list[0]._dict.keys() == qubit_pauli_operator_noisy._dict.keys()
    assert qubit_pauli_operator_list[1]._dict.keys() == qubit_pauli_operator_noisless._dict.keys()

    assert abs(qubit_pauli_operator_list[0]._dict[qps_noisy_noisy] - 0.3439) < 0.1
    assert abs(qubit_pauli_operator_list[0]._dict[qps_noisy_noisless]) < 0.1
    assert abs(qubit_pauli_operator_list[1]._dict[qps_noisless_one] - 1) < 0.1
    assert abs(qubit_pauli_operator_list[1]._dict[qps_noisless_zero]) < 0.1


@pytest.mark.high_compute
def test_end_to_end_noise_aware_zne_mitex_starting_from_ptm() -> None:

    mp.set_start_method("spawn", force=True)

    # Here we are creating the PTM for a noise model acting
    # XI with rate 0.1
    ptm = np.diag([1, 1, 1, 1, 1, 1, 1, 1, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8])
    pauli_index = {pauli: index for index, pauli in enumerate(product([Pauli.I, Pauli.X, Pauli.Y, Pauli.Z], repeat=2))}
    error_distribution = ErrorDistribution.from_ptm(ptm, pauli_index)

    noise_model = NoiseModel(
        noise_model={
            OpType.CZ: error_distribution
        }
    )
    transpiler = PauliErrorTranspile(noise_model=noise_model)
    backend = TranspilerBackend(transpiler=transpiler)

    # Here we perform ZNE with some unevenly spaced
    # noise scaling values.
    zne_mitex = gen_ZNE_MitEx(
        backend=backend,
        noise_scaling_list=[3.6, 1, 4, 2, 2.5, 3, 3.3],
        folding_type=Folding.noise_aware,
        fit_type=Fit.exponential,
        noise_model=noise_model,
        n_noisy_circuit_samples=1000,
    )

    circuit_noisy = Circuit(2).CZ(0, 1)

    qps_noisy_noisy = QubitPauliString(map={Qubit(0): Pauli.Z, Qubit(1): Pauli.Z})

    qubit_pauli_operator_noisy = QubitPauliOperator(
        dictionary={qps_noisy_noisy: 1}
    )

    observable_tracker_noisy = ObservableTracker(
        qubit_pauli_operator=qubit_pauli_operator_noisy
    )

    shots = 10000
    ansatz_circuit_noisy = AnsatzCircuit(
        Circuit=circuit_noisy,
        Shots=shots,
        SymbolsDict={}
    )

    observable_experiment_noisy = ObservableExperiment(
        AnsatzCircuit=ansatz_circuit_noisy,
        ObservableTracker=observable_tracker_noisy,
    )

    qubit_pauli_operator_list = zne_mitex.run(
        mitex_wires=[observable_experiment_noisy]
    )

    assert abs(qubit_pauli_operator_list[0]._dict[qps_noisy_noisy] - 1) < 0.1


@pytest.mark.high_compute
def test_end_to_end_noise_aware_zne_mitex():

    mp.set_start_method("spawn", force=True)

    error_rate = 0.1
    error_distribution = ErrorDistribution(
        distribution={(Pauli.X, Pauli.I): error_rate}
    )
    noise_model = NoiseModel(
        noise_model={OpType.CZ: error_distribution}
    )
    transpiler = PauliErrorTranspile(noise_model=noise_model)
    backend = TranspilerBackend(transpiler=transpiler)

    zne_mitex = gen_ZNE_MitEx(
        backend=backend,
        noise_scaling_list=[2, 5, 3, 1, 4],
        folding_type=Folding.noise_aware,
        fit_type=Fit.exponential,
        noise_model=noise_model,
        n_noisy_circuit_samples=1000,
    )

    circuit_noisy = Circuit(2).CZ(0, 1)

    qps_noisy_noisy = QubitPauliString(map={Qubit(0): Pauli.Z, Qubit(1): Pauli.Z})

    qubit_pauli_operator_noisy = QubitPauliOperator(
        dictionary={qps_noisy_noisy: 1}
    )

    observable_tracker_noisy = ObservableTracker(
        qubit_pauli_operator=qubit_pauli_operator_noisy
    )

    shots = 10000
    ansatz_circuit_noisy = AnsatzCircuit(
        Circuit=circuit_noisy,
        Shots=shots,
        SymbolsDict={}
    )

    observable_experiment_noisy = ObservableExperiment(
        AnsatzCircuit=ansatz_circuit_noisy,
        ObservableTracker=observable_tracker_noisy,
    )

    qubit_pauli_operator_list = zne_mitex.run(
        mitex_wires=[observable_experiment_noisy]
    )

    assert abs(qubit_pauli_operator_list[0]._dict[qps_noisy_noisy] - 1) < 0.1


def test_noise_aware_folding():

    error_distribution = ErrorDistribution(
        distribution={(Pauli.X, Pauli.I): 1}
    )
    noise_model = NoiseModel(
        noise_model={OpType.CZ: error_distribution}
    )
    circ = Circuit(2).CZ(0, 1)
    scaled_circ = Folding.noise_aware(
        circ=circ,
        noise_scaling=2,
        noise_model=noise_model,
        n_noisy_circuit_samples=1,
    )
    assert scaled_circ[0] == Circuit(2).CZ(0, 1).X(0, opgroup='noisy')


def test_merge_experiments_task_gen():

    task = merge_experiments_task_gen()

    qps_one = QubitPauliString(map={Qubit(0): Pauli.Z})
    qps_two = QubitPauliString(map={Qubit(1): Pauli.X})

    qpo_one = QubitPauliOperator(
        dictionary={qps_one: 1, qps_two: 1}
    )
    qpo_two = QubitPauliOperator(
        dictionary={qps_one: 1, qps_two: -1}
    )
    qpo_three = QubitPauliOperator(
        dictionary={qps_one: 1, qps_two: -1}
    )
    merged_qpo_list = task(
        ([qpo_one, qpo_two, qpo_three], [0, 0, 1])
    )
    assert merged_qpo_list[0][0]._dict[qps_one] == 1
    assert merged_qpo_list[0][0]._dict[qps_two] == 0
    assert merged_qpo_list[0][1]._dict[qps_one] == 1
    assert merged_qpo_list[0][1]._dict[qps_two] == -1


if __name__ == "__main__":
    test_no_qubit_relabel()
    test_extrapolation_task_gen()
    test_gen_duplication_task()
    test_digital_folding_task_gen()
    test_gen_initial_compilation_task()
    test_zne_identity()
    test_simple_run_end_to_end()
    test_odd_gate_folding()
    test_circuit_folding_TK1()
    test_gen_qubit_relabel_task()
    test_two_qubit_gate_folding()
    test_gen_initial_compilation_task_quantinuum_qubit_names()
    test_end_to_end_noise_scaled_mitex()
    test_end_to_end_noise_aware_zne_mitex()
    test_end_to_end_noise_aware_zne_mitex_starting_from_ptm()
    test_merge_experiments_task_gen()
    test_noise_aware_folding()

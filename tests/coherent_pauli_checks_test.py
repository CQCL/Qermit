from pytket import Circuit
from qermit.coherent_pauli_checks import (
    QermitDAGCircuit,
    cpc_rebase_pass,
    DeterministicZPauliSampler,
    DeterministicXPauliSampler,
    RandomPauliSampler,
    OptimalPauliSampler,
)
from pytket.circuit import Qubit, Bit
from qermit.probabilistic_error_cancellation.cliff_circuit_gen import random_clifford_circ
import pytest
from pytket.passes import DecomposeBoxes
from quantinuum_benchmarking.noise_model import (
    ErrorDistribution,
    NoiseModel,
)
from pytket.pauli import Pauli
from pytket import OpType
from quantinuum_benchmarking.direct_fidelity_estimation import Stabiliser
from pytket.circuit import Bit
from pytket.extensions.qiskit import AerBackend


def test_get_clifford_subcircuits():

    circ = Circuit(3).CZ(0, 1).H(1).Z(1).CZ(1, 0)
    cliff_circ = QermitDAGCircuit(circ)
    assert cliff_circ.get_clifford_subcircuits() == [0, 0, 0, 0]

    circ = Circuit(3).CZ(1, 2).H(2).Z(1).CZ(0, 1).H(1).CZ(1, 0).Z(1).CZ(1, 2)
    cliff_circ = QermitDAGCircuit(circ)
    assert cliff_circ.get_clifford_subcircuits() == [0, 0, 0, 0, 0, 0, 0, 0]

    circ = Circuit(3).CZ(1, 2).Rz(0.1, 1).H(2).Z(1).CZ(0, 1).Rz(0.1, 1).CZ(1, 0).Z(1).CZ(1, 2)
    cliff_circ = QermitDAGCircuit(circ)
    assert cliff_circ.get_clifford_subcircuits() == [0, 1, 0, 2, 2, 3, 4, 4, 4]


def test_add_pauli_checks():

    circ = Circuit(3).H(1).CX(1, 0)
    cpc_rebase_pass.apply(circ)
    cliff_circ = QermitDAGCircuit(circ)
    circuit = cliff_circ.add_pauli_checks(pauli_sampler=DeterministicZPauliSampler())

    ideal_circ = Circuit(3)

    ancilla_0 = Qubit(name='ancilla', index=0)
    ancilla_1 = Qubit(name='ancilla', index=1)

    ancilla_measure_0 = Bit(name='ancilla_measure', index=0)
    ancilla_measure_1 = Bit(name='ancilla_measure', index=1)

    ideal_circ.add_qubit(ancilla_0)
    ideal_circ.add_qubit(ancilla_1)

    ideal_circ.add_bit(id=ancilla_measure_0)
    ideal_circ.add_bit(id=ancilla_measure_1)

    ideal_circ.add_barrier([ideal_circ.qubits[3], ancilla_0])
    ideal_circ.H(ancilla_0)
    ideal_circ.CZ(control_qubit=ancilla_0, target_qubit=ideal_circ.qubits[3])
    ideal_circ.add_barrier([ideal_circ.qubits[3], ancilla_0])

    ideal_circ.H(ideal_circ.qubits[3])

    ideal_circ.add_barrier([ideal_circ.qubits[3], ancilla_0])
    ideal_circ.CX(control_qubit=ancilla_0, target_qubit=ideal_circ.qubits[3])
    ideal_circ.H(ancilla_0)
    ideal_circ.add_barrier([ideal_circ.qubits[3], ancilla_0])

    ideal_circ.add_barrier([ideal_circ.qubits[3], ideal_circ.qubits[2], ancilla_1])
    ideal_circ.H(ancilla_1)
    ideal_circ.CZ(control_qubit=ancilla_1, target_qubit=ideal_circ.qubits[3])
    ideal_circ.CZ(control_qubit=ancilla_1, target_qubit=ideal_circ.qubits[2])
    ideal_circ.add_barrier([ideal_circ.qubits[3], ideal_circ.qubits[2], ancilla_1])

    ideal_circ.H(ideal_circ.qubits[2])
    ideal_circ.CZ(control_qubit=ideal_circ.qubits[3], target_qubit=ideal_circ.qubits[2])
    ideal_circ.H(ideal_circ.qubits[2])

    ideal_circ.add_barrier([ideal_circ.qubits[3], ideal_circ.qubits[2], ancilla_1])
    ideal_circ.CZ(control_qubit=ancilla_1, target_qubit=ideal_circ.qubits[2])
    ideal_circ.H(ancilla_1)
    ideal_circ.add_barrier([ideal_circ.qubits[3], ideal_circ.qubits[2], ancilla_1])

    ideal_circ.Measure(ancilla_0, ancilla_measure_0)
    ideal_circ.Measure(ancilla_1, ancilla_measure_1)

    assert ideal_circ == circuit

    circ = Circuit(2).H(0).CX(1, 0).X(1).CX(1, 0)
    cpc_rebase_pass.apply(circ)
    cliff_circ = QermitDAGCircuit(circ)
    circuit = cliff_circ.add_pauli_checks(pauli_sampler=DeterministicZPauliSampler())

    ideal_circ = Circuit(2)

    ancilla = Qubit(name='ancilla', index=0)
    ancilla_measure = Bit(name='ancilla_measure', index=0)

    ideal_circ.add_qubit(ancilla)
    ideal_circ.add_bit(id=ancilla_measure)

    ideal_circ.add_barrier([ideal_circ.qubits[2], ideal_circ.qubits[1], ancilla])
    ideal_circ.H(ancilla)
    ideal_circ.CZ(control_qubit=ancilla, target_qubit=ideal_circ.qubits[2])
    ideal_circ.CZ(control_qubit=ancilla, target_qubit=ideal_circ.qubits[1])
    ideal_circ.add_barrier([ideal_circ.qubits[2], ideal_circ.qubits[1], ancilla])

    ideal_circ.H(ideal_circ.qubits[1])
    ideal_circ.H(ideal_circ.qubits[1])
    ideal_circ.CZ(control_qubit=ideal_circ.qubits[2], target_qubit=ideal_circ.qubits[1])
    ideal_circ.H(ideal_circ.qubits[1])
    ideal_circ.H(ideal_circ.qubits[1])
    ideal_circ.X(ideal_circ.qubits[2])
    ideal_circ.CZ(control_qubit=ideal_circ.qubits[2], target_qubit=ideal_circ.qubits[1])
    ideal_circ.H(ideal_circ.qubits[1])

    ideal_circ.add_barrier([ideal_circ.qubits[2], ideal_circ.qubits[1], ancilla])
    ideal_circ.CZ(control_qubit=ancilla, target_qubit=ideal_circ.qubits[2])
    ideal_circ.CX(control_qubit=ancilla, target_qubit=ideal_circ.qubits[1])
    ideal_circ.S(ancilla)
    ideal_circ.S(ancilla)
    ideal_circ.H(ancilla)
    ideal_circ.add_barrier([ideal_circ.qubits[2], ideal_circ.qubits[1], ancilla])

    ideal_circ.Measure(ancilla, ancilla_measure)

    assert ideal_circ == circuit


def test_simple_non_minimal_example():

    # Note that this is a simple example of where the current implementation
    # is not minimal. The whole think is a relatively easy to identify
    # Clifford circuit.

    clifford_circuit = Circuit(3).CZ(0, 1).X(2).X(0).CZ(0, 2).CZ(1, 2)
    dag_circuit = QermitDAGCircuit(clifford_circuit)
    assert dag_circuit.get_clifford_subcircuits() == [0, 1, 0, 1, 1]


def test_5q_random_clifford():

    clifford_circuit = random_clifford_circ(n_qubits=5, seed=0)
    cpc_rebase_pass.apply(clifford_circuit)
    dag_circuit = QermitDAGCircuit(clifford_circuit)
    pauli_sampler = RandomPauliSampler(seed=0)
    dag_circuit.add_pauli_checks(pauli_sampler=pauli_sampler)


@pytest.mark.skip(reason="This test passes, but the functionality is incorrect. In particular there is a H in the middle which is identified as Clifford but which has no checks added.")
def test_2q_random_clifford():

    clifford_circuit = random_clifford_circ(n_qubits=5, seed=0)
    cpc_rebase_pass.apply(clifford_circuit)
    dag_circuit = QermitDAGCircuit(clifford_circuit)
    pauli_sampler = RandomPauliSampler(seed=0)
    dag_circuit.add_pauli_checks(pauli_sampler=pauli_sampler)


def test_CZ_circuit_with_phase():

    # This test is a case where the pauli circuit to be controlled has a
    # global phase which needs to be bumped to the control.

    original_circuit = Circuit(2).CZ(0, 1).measure_all()
    dag_circuit = QermitDAGCircuit(original_circuit)
    pauli_sampler = DeterministicXPauliSampler()
    pauli_checks_circuit = dag_circuit.add_pauli_checks(pauli_sampler=pauli_sampler)

    ideal_circ = Circuit(2, 2)
    comp_qubits = ideal_circ.qubits
    comp_bits = ideal_circ.bits

    ancilla = Qubit(name='ancilla', index=0)
    ancilla_measure = Bit(name='ancilla_measure', index=0)

    ideal_circ.add_qubit(ancilla)
    ideal_circ.add_bit(id=ancilla_measure)

    ideal_circ.add_barrier([*list(reversed(comp_qubits)), ancilla])
    ideal_circ.H(ancilla)
    ideal_circ.CX(ancilla, comp_qubits[1])
    ideal_circ.CX(ancilla, comp_qubits[0])
    ideal_circ.add_barrier([*list(reversed(comp_qubits)), ancilla])
    ideal_circ.CZ(comp_qubits[0], comp_qubits[1])
    ideal_circ.add_barrier([*list(reversed(comp_qubits)), ancilla])
    ideal_circ.CZ(ancilla, comp_qubits[1])
    ideal_circ.CX(ancilla, comp_qubits[1])
    ideal_circ.CZ(ancilla, comp_qubits[0])
    ideal_circ.CX(ancilla, comp_qubits[0])
    ideal_circ.S(ancilla)
    ideal_circ.S(ancilla)
    ideal_circ.H(ancilla)
    ideal_circ.add_barrier([*list(reversed(comp_qubits)), ancilla])
    ideal_circ.Measure(comp_qubits[0], comp_bits[0])
    ideal_circ.Measure(comp_qubits[1], comp_bits[1])
    ideal_circ.Measure(ancilla, ancilla_measure)

    assert pauli_checks_circuit == ideal_circ

def test_to_clifford_subcircuits():

    orig_circuit = Circuit(3).CZ(1, 2).Rz(0.1, 1).H(2).Z(1).CZ(0, 1).Rz(0.1, 1).CZ(1, 0).Z(1).CZ(1, 2)
    dag_circuit = QermitDAGCircuit(orig_circuit)
    clifford_box_circuit = dag_circuit.to_clifford_subcircuit_boxes()
    DecomposeBoxes().apply(clifford_box_circuit)
    assert clifford_box_circuit == orig_circuit

def test_optimal_pauli_sampler():

    # TODO: add a measure and barrier to this circuit, just to check
    cliff_circ = Circuit()
    cliff_circ.add_q_register(name='my_reg', size=3)
    qubits = cliff_circ.qubits
    cliff_circ.CZ(qubits[0],qubits[1]).CZ(qubits[1],qubits[2])

    error_distribution_dict = {}
    error_distribution_dict[(Pauli.X, Pauli.I)] = 0.3
    error_distribution_dict[(Pauli.I, Pauli.X)] = 0.7

    error_distribution = ErrorDistribution(error_distribution_dict, seed=0)
    noise_model = NoiseModel({OpType.CZ:error_distribution})

    pauli_sampler = OptimalPauliSampler(noise_model)
    stab = pauli_sampler.sample(cliff_circ.qubits, cliff_circ)

    assert stab[0] == Stabiliser(
        Z_list=[0,0,1],
        X_list=[0,0,1],
        qubit_list=qubits,
        phase=1,
    )

    # TODO: an assert is needed for this last part

    pauli_sampler = OptimalPauliSampler(noise_model)
    dag_circ = QermitDAGCircuit(cliff_circ)
    pauli_check_circ = dag_circ.add_pauli_checks(pauli_sampler=pauli_sampler)

def test_add_ZX_pauli_checks_to_S():

    cliff_circ = Circuit()
    cliff_circ.add_q_register(name='my_reg', size=1)
    qubits = cliff_circ.qubits
    cliff_circ.S(qubits[0])
    cliff_circ.measure_all()

    class DeterministicPauliSampler:

        def sample(self, qubit_list, **kwargs):
            return [Stabiliser(
                Z_list=[1],
                X_list=[1],
                qubit_list=qubits,
            )]
    
    dag_circ = QermitDAGCircuit(cliff_circ)
    pauli_sampler = DeterministicPauliSampler()
    pauli_check_circ = dag_circ.add_pauli_checks(
        pauli_sampler=pauli_sampler,
        n_rand=10000,
        cutoff=10,
    )

    ideal_circ = Circuit()

    ancilla = Qubit(name='ancilla', index=0)
    comp = Qubit(name='my_reg', index=0)
    ancilla_measure = Bit(name='ancilla_measure', index=0)
    comp_measure = Bit(name='c', index=0)

    ideal_circ.add_qubit(ancilla)
    ideal_circ.add_qubit(comp)

    ideal_circ.add_bit(id=ancilla_measure)
    ideal_circ.add_bit(id=comp_measure)

    ideal_circ.add_barrier([comp, ancilla]).H(ancilla).CZ(ancilla, comp).CX(ancilla, comp).add_barrier([comp, ancilla])
    ideal_circ.S(comp)
    ideal_circ.add_barrier([comp, ancilla]).CX(ancilla, comp).S(ancilla).S(ancilla).S(ancilla).H(ancilla).add_barrier([comp, ancilla])
    ideal_circ.Measure(ancilla, ancilla_measure)
    ideal_circ.Measure(comp, comp_measure)

    assert ideal_circ == pauli_check_circ

    backend = AerBackend()
    backend.rebase_pass().apply(pauli_check_circ)
    result=backend.run_circuit(pauli_check_circ, n_shots=100)
    counts = result.get_counts()

    assert list(counts.keys()) == [(0,0)]
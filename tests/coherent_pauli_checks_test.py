from pytket import Circuit
from qermit.coherent_pauli_checks import QermitDAGCircuit, cpc_rebase_pass
from qermit.coherent_pauli_checks import DeterministicZPauliSampler
from pytket.circuit import Qubit, Bit


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
    ideal_circ.CZ(control_qubit=ancilla_0, target_qubit=ideal_circ.qubits[3])
    ideal_circ.add_barrier([ideal_circ.qubits[3], ancilla_0])
    ideal_circ.H(ideal_circ.qubits[3])
    ideal_circ.add_barrier([ideal_circ.qubits[3], ancilla_0])
    ideal_circ.CX(control_qubit=ancilla_0, target_qubit=ideal_circ.qubits[3])
    ideal_circ.add_barrier([ideal_circ.qubits[3], ancilla_0])
    ideal_circ.add_barrier([ideal_circ.qubits[3], ideal_circ.qubits[2], ancilla_1])
    ideal_circ.CZ(control_qubit=ancilla_1, target_qubit=ideal_circ.qubits[3])
    ideal_circ.CZ(control_qubit=ancilla_1, target_qubit=ideal_circ.qubits[2])
    ideal_circ.add_barrier([ideal_circ.qubits[3], ideal_circ.qubits[2], ancilla_1])
    ideal_circ.H(ideal_circ.qubits[2])
    ideal_circ.CZ(control_qubit=ideal_circ.qubits[3], target_qubit=ideal_circ.qubits[2])
    ideal_circ.H(ideal_circ.qubits[2])
    ideal_circ.add_barrier([ideal_circ.qubits[3], ideal_circ.qubits[2], ancilla_1])
    ideal_circ.CZ(control_qubit=ancilla_1, target_qubit=ideal_circ.qubits[2])
    ideal_circ.add_barrier([ideal_circ.qubits[3], ideal_circ.qubits[2], ancilla_1])
    ideal_circ.Measure(ancilla_0, ancilla_measure_0)
    ideal_circ.Measure(ancilla_1, ancilla_measure_1)

    assert ideal_circ == circuit

    circ = Circuit(2).H(0).CX(1,0).X(1).CX(1,0)
    cpc_rebase_pass.apply(circ)
    cliff_circ = QermitDAGCircuit(circ)
    circuit = cliff_circ.add_pauli_checks(pauli_sampler=DeterministicZPauliSampler())

    ideal_circ = Circuit(2)

    ancilla = Qubit(name='ancilla', index=0)
    ancilla_measure = Bit(name='ancilla_measure', index=0)

    ideal_circ.add_qubit(ancilla)
    ideal_circ.add_bit(id=ancilla_measure)

    ideal_circ.add_barrier([ideal_circ.qubits[2], ideal_circ.qubits[1], ancilla])
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
    ideal_circ.add_barrier([ideal_circ.qubits[2], ideal_circ.qubits[1], ancilla])
    ideal_circ.add_phase(1)
    ideal_circ.Measure(ancilla, ancilla_measure)

    assert ideal_circ == circuit

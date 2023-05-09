from pytket import Circuit
from qermit.coherent_pauli_checks import QermitDAGCircuit
import networkx as nx


def test_get_clifford_subcircuits():

    circ = Circuit(3).CX(0,1).H(1).X(1).CX(1,0)
    cliff_circ = QermitDAGCircuit(circ)
    assert cliff_circ.get_clifford_subcircuits() == [0, 0, None, 1]

    circ = Circuit(3).CX(1,2).H(2).X(1).CX(0,1).H(1).CX(1,0).X(1).CX(1,2)
    cliff_circ = QermitDAGCircuit(circ)
    assert cliff_circ.get_clifford_subcircuits() == [0, None, 0, 1, 1, 1, None, 2]

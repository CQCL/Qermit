from pytket import Circuit, OpType
from pytket.utils import Graph
import networkx as nx
from pytket.circuit import Command

clifford_ops = [OpType.CX, OpType.H]

class DAGCommand:

    def __init__(self, command:Command):

        self.command = command
        self.clifford = command.op.type in clifford_ops

class QermitDAGCircuit(nx.DiGraph):

    def __init__(self, circuit:Circuit):

        # TODO: There are other things to be saved, like the phase.

        super().__init__()
        
        self.node_command = [
            DAGCommand(command) for command in circuit.get_commands()
        ]
        self.qubits = circuit.qubits

        current_node = {}
        for node, command in enumerate(self.node_command):
            self.add_node(node)
            for qubit in command.command.qubits:
                if qubit in current_node.keys():
                    self.add_edge(current_node[qubit], node)
                current_node[qubit] = node

    def get_clifford_subcircuits(self):

        node_sub_circuit = [None for _ in self.nodes]
        sub_circuit_count = 0

        for node, command in enumerate(self.node_command):

            if not command.clifford:
                continue

            if node_sub_circuit[node] is None:
                node_sub_circuit[node] = sub_circuit_count
                sub_circuit_count += 1

            for neighbour_id in self.neighbors(node):
                
                if not self.node_command[neighbour_id].clifford:
                    continue

                # TODO: This should be removed. In particular we can include
                # the current node in the hyperedge if there are no
                # non-clifford blocking us from doing so.
                if node_sub_circuit[neighbour_id] is not None:
                    continue

                same_sub_circuit_node_list = [
                    i for i, sub_circuit in enumerate(node_sub_circuit)
                    if sub_circuit == node_sub_circuit[node]
                ]

                same_clifford_circuit = True
                for same_sub_circuit_node in same_sub_circuit_node_list:

                    for path in nx.all_simple_paths(
                        self, same_sub_circuit_node, neighbour_id
                    ):
                        if not all(
                            self.node_command[path_node].clifford
                            for path_node in path
                        ):
                            same_clifford_circuit = False

                if same_clifford_circuit:
                    node_sub_circuit[neighbour_id] = node_sub_circuit[node]
        
        return node_sub_circuit

    def get_sub_circuit_qubits(self, node_sub_circuit):

        sub_circuit_list = list(set(node_sub_circuit))
        sub_circuit_qubits = {
            sub_circuit:set() for sub_circuit in sub_circuit_list
        }
        for node, sub_circuit in enumerate(node_sub_circuit):
            for qubit in self.node_command[node].command.qubits:
                sub_circuit_qubits[sub_circuit].add(qubit)

        return sub_circuit_qubits

    def add_pauli_checks(self):

        node_sub_circuit_list = self.get_clifford_subcircuits()
        sub_circuit_qubits = self.get_sub_circuit_qubits(node_sub_circuit_list)
        node_sub_circuit_list = [
            (i, node_sub_circuit)
            for i, node_sub_circuit
            in enumerate(node_sub_circuit_list)
        ]
        
        pauli_check_circuit = Circuit()
        for quibt in self.qubits:
            pauli_check_circuit.add_qubit(quibt)

        while node_sub_circuit_list:

            first_node_sub_circuit = node_sub_circuit_list.pop(0)

            if first_node_sub_circuit[1] is not None:
                pauli_check_circuit.add_barrier(
                    list(sub_circuit_qubits[first_node_sub_circuit[1]])
                )

            pauli_check_circuit.add_gate(
                self.node_command[first_node_sub_circuit[0]].command.op,
                self.node_command[first_node_sub_circuit[0]].command.qubits
            )

            if first_node_sub_circuit[1] is None:   
                continue

            for node_sub_circuit in node_sub_circuit_list:
                
                if not node_sub_circuit[1] == first_node_sub_circuit[1]:
                    continue

                pauli_check_circuit.add_gate(
                    self.node_command[node_sub_circuit[0]].command.op,
                    self.node_command[node_sub_circuit[0]].command.qubits
                )

            node_sub_circuit_list = [
                node_sub_circuit
                for node_sub_circuit in node_sub_circuit_list
                if node_sub_circuit[1] != first_node_sub_circuit[1]
            ]

            pauli_check_circuit.add_barrier(
                list(sub_circuit_qubits[first_node_sub_circuit[1]])
            )

        return pauli_check_circuit

            
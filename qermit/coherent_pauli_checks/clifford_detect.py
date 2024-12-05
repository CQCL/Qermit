import math
from typing import List, Optional, Union, cast

import networkx as nx  # type: ignore
from pytket import Circuit, OpType, Qubit
from pytket.circuit import CircBox, Command
from pytket.passes import AutoRebase

from .monochromatic_convex_subdag import MonochromaticConvexSubDAG

clifford_ops = [OpType.CZ, OpType.H, OpType.Z, OpType.S, OpType.X]
non_clifford_ops = [OpType.Rz]

cpc_rebase_pass = AutoRebase(gateset=set(clifford_ops + non_clifford_ops))


class DAGCommand:
    def __init__(self, command: Command) -> None:
        self.command = command

    @property
    def clifford(self) -> bool:
        if self.command.op.is_clifford_type():
            return True

        if self.command.op.type in [OpType.PhasedX, OpType.Rz]:
            return all(
                math.isclose(param % 0.5, 0) or math.isclose(param % 0.5, 0.5)
                for param in self.command.op.params
            )

        return False


class QermitDAGCircuit:
    def __init__(self, circuit: Circuit) -> None:
        # TODO: There are other things to be saved, like the phase.

        self.node_command = [command for command in circuit.get_commands()]
        self.qubits = circuit.qubits
        self.bits = circuit.bits

        # Lists the most recent node to act on a particular qubits. If a
        # new gate is found to act on that qubit then an edge between the
        # node which corresponds to the new gate and the node which
        # most recently acted on that qubit is added. This builds a DAG
        # showing the temporal order of gates.
        current_node: dict[Qubit, int] = {}
        self.dag = nx.DiGraph()
        for node, command in enumerate(self.node_command):
            self.dag.add_node(node)
            for qubit in command.qubits:
                # This if statement is used in case the qubit has not been
                # acted on yet.
                if qubit in current_node.keys():
                    self.dag.add_edge(current_node[qubit], node)
                current_node[qubit] = node

        def command_is_clifford(command):
            if command.op.is_clifford_type():
                return True

            if command.op.type in [OpType.PhasedX, OpType.Rz]:
                return all(
                    math.isclose(param % 0.5, 0) or math.isclose(param % 0.5, 0.5)
                    for param in command.op.params
                )

            return False

        self.node_clifford = {
            node: command_is_clifford(command)
            for node, command in enumerate(self.node_command)
        }

        self.monochromatic_convex_subdag = MonochromaticConvexSubDAG(
            dag=self.dag,
            node_coloured=self.node_clifford,
        )

    def get_clifford_subcircuits(self) -> List[int]:
        node_subdag = self.monochromatic_convex_subdag.greedy_merge()
        node_sub_circuit_list = []
        sub_circuit_index = 0
        for node, command in enumerate(self.node_command):
            if node in node_subdag.keys():
                node_sub_circuit_list.append(node_subdag[node])
            else:
                while (sub_circuit_index in node_subdag.values()) or (
                    sub_circuit_index in node_sub_circuit_list
                ):
                    sub_circuit_index += 1
                node_sub_circuit_list.append(sub_circuit_index)

        return node_sub_circuit_list

    # TODO: I'm not sure if this should return a circuit, or changes this
    # QermitDagCircuit in place
    def to_clifford_subcircuit_boxes(self) -> Circuit:
        # TODO: It could be worth insisting that the given circuit does not
        # include any boxes called 'Clifford Subcircuit'. i.e. that the
        # circuit is 'clean'.

        node_sub_circuit_list = self.get_clifford_subcircuits()
        sub_circuit_qubits = self.get_sub_circuit_qubits(node_sub_circuit_list)

        # List indicating if a command has been implemented
        implemented_commands = [False for _ in self.dag.nodes]

        # Initialise new circuit
        clifford_box_circuit = Circuit()
        for qubit in self.qubits:
            clifford_box_circuit.add_qubit(qubit)
        for bit in self.bits:
            clifford_box_circuit.add_bit(bit)

        while not all(implemented_commands):
            # Search for a subcircuit that it is safe to implement, and
            # pick the first one found to be implemented.
            not_implemented = [
                node_sub_circuit
                for node_sub_circuit, implemented in zip(
                    node_sub_circuit_list, implemented_commands
                )
                if not implemented
            ]
            sub_circuit_to_implement = None
            for sub_circuit in set(not_implemented):
                if self.can_implement(
                    sub_circuit, node_sub_circuit_list, implemented_commands
                ):
                    sub_circuit_to_implement = sub_circuit
                    break
            assert sub_circuit_to_implement is not None

            # List the nodes in the chosen sub circuit
            node_to_implement_list = [
                node
                for node, _ in enumerate(self.dag.nodes)
                if node_sub_circuit_list[node] == sub_circuit_to_implement
            ]
            assert len(node_to_implement_list) > 0

            # If the circuit is clifford add it as a circbox
            # if self.node_command[node_to_implement_list[0]].clifford:
            if self.node_clifford[node_to_implement_list[0]]:
                # Empty circuit to contain clifford subcircuit
                clifford_subcircuit = Circuit(
                    n_qubits=len(sub_circuit_qubits[sub_circuit_to_implement]),
                    name="Clifford Subcircuit",
                )

                # Map from qubits in original circuit to qubits in new
                # clifford circuit.
                qubit_to_index = {
                    qubit: i
                    for i, qubit in enumerate(
                        sub_circuit_qubits[sub_circuit_to_implement]
                    )
                }

                # Add all gates to new circuit
                for node in node_to_implement_list:
                    clifford_subcircuit.add_gate(
                        self.node_command[node].op,
                        [
                            qubit_to_index[qubit]
                            for qubit in self.node_command[node].args
                        ],
                    )
                    implemented_commands[node] = True

                clifford_circ_box = CircBox(clifford_subcircuit)
                clifford_box_circuit.add_circbox(
                    clifford_circ_box,
                    list(sub_circuit_qubits[sub_circuit_to_implement]),
                )

            # Otherwise, add the gates straight to the circuit
            else:
                for node in node_to_implement_list:
                    clifford_box_circuit.add_gate(
                        self.node_command[node].op,
                        self.node_command[node].args,
                    )
                    implemented_commands[node] = True

        return clifford_box_circuit

    def get_sub_circuit_qubits(
        self, node_sub_circuit: list[int]
    ) -> dict[int, set[Qubit]]:
        """Creates a dictionary from the clifford sub circuit to the qubits
        which it covers.

        :param node_sub_circuit: List identifying to which clifford sub circuit
            each command belongs.
        :type node_sub_circuit: List[Int]
        :return: Dictionary from clifford sub circuit index to the qubits
            the circuit covers.
        :rtype: Dict[Int, List[Quibt]]
        """

        sub_circuit_list = list(set(node_sub_circuit))
        sub_circuit_qubits: dict[int, set[Qubit]] = {
            sub_circuit: set() for sub_circuit in sub_circuit_list
        }
        for node, sub_circuit in enumerate(node_sub_circuit):
            for qubit in self.node_command[node].qubits:
                sub_circuit_qubits[sub_circuit].add(qubit)

        return sub_circuit_qubits

    def can_implement(
        self,
        sub_circuit: int,
        node_sub_circuit_list: List[int],
        implemented_commands: List[bool],
    ) -> bool:
        can_implement = True
        for node, _ in enumerate(self.dag.nodes):
            if not node_sub_circuit_list[node] == sub_circuit:
                continue

            for predecessor in self.dag.predecessors(node):
                if node_sub_circuit_list[predecessor] == sub_circuit:
                    continue
                if not implemented_commands[predecessor]:
                    can_implement = False

        return can_implement

from pytket import Circuit, OpType, Qubit
import networkx as nx  # type: ignore
from pytket.circuit import Command, CircBox  # type: ignore
from pytket.passes.auto_rebase import auto_rebase_pass
import math
from typing import Optional, List, Union, cast


clifford_ops = [OpType.CZ, OpType.H, OpType.Z, OpType.S, OpType.X]
non_clifford_ops = [OpType.Rz]

cpc_rebase_pass = auto_rebase_pass(
    gateset=set(clifford_ops + non_clifford_ops)
)


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


class QermitDAGCircuit(nx.DiGraph):

    def __init__(self, circuit: Circuit, cutoff: Optional[int] = None) -> None:

        # TODO: There are other things to be saved, like the phase.

        super().__init__()

        self.node_command = [
            DAGCommand(command) for command in circuit.get_commands()
        ]
        self.qubits = circuit.qubits
        self.bits = circuit.bits
        self.cutoff = cutoff

        # Lists the most recent node to act on a particular qubits. If a
        # new gate is found to act on that qubit then an edge between the
        # node which corresponds to the new gate and the node which
        # most recently acted on that qubit is added. This builds a DAG
        # showing the temporal order of gates.
        current_node: dict[Qubit, int] = {}
        for node, command in enumerate(self.node_command):
            self.add_node(node)
            for qubit in command.command.qubits:
                # This if statement is used in case the qubit has not been
                # acted on yet.
                if qubit in current_node.keys():
                    self.add_edge(current_node[qubit], node)
                current_node[qubit] = node

    def get_clifford_subcircuits(self) -> List[int]:

        # a list indicating the clifford subcircuit to which a command belongs.
        node_sub_circuit: List[Union[int, None]] = [None] * self.number_of_nodes()
        next_sub_circuit_id = 0

        # Iterate through all commands and check if their neighbours should
        # be added to the same clifford subcircuit.
        for node, command in enumerate(self.node_command):

            # If the command is not in a clifford sub circuit, start a
            # new one and add it to that new one,
            if node_sub_circuit[node] is None:
                node_sub_circuit[node] = next_sub_circuit_id
                next_sub_circuit_id += 1

            # Ignore the command if it is not clifford
            if not command.clifford:
                continue

            # For all the neighbours of the command being considered, add that
            # neighbour to the same clifford subcircuit if no non-clifford
            # gates prevent this from happening.
            for neighbour_id in self.neighbors(node):

                # Do not add the neighbour if it is not a clifford gate.
                if not self.node_command[neighbour_id].clifford:
                    continue

                # Do not add the neighbour if it is already part of a clifford
                # sub circuit.
                # TODO: This should be removed. In particular we can include
                # the current node in the hyperedge if there are no
                # non-clifford blocking us from doing so.
                if node_sub_circuit[neighbour_id] is not None:
                    continue

                # list all of the commands in the circuit which belong to
                # the same sub circuit as the one being considered
                same_sub_circuit_node_list = [
                    i for i, sub_circuit in enumerate(node_sub_circuit)
                    if sub_circuit == node_sub_circuit[node]
                ]

                # Check if any of the paths in the circuit from the neighbour
                # to other commands in the clifford circuit pass through
                # a different Clifford subcircuit. If nodes on the path
                # belonged to another clifford subcircuit then it would
                # not be possible to build the circuit by applying
                # sub circuits sequentially.
                same_clifford_circuit = True
                for same_sub_circuit_node in same_sub_circuit_node_list:

                    # I'm allowing to pass cutoff, but that should
                    # not be allowed. In particular paths of arbitrary
                    # lengths should be checked in practice.
                    # however all_simple_paths is quite slow otherwise as it
                    # spends a lot of time looking for paths that don't exist.
                    for path in nx.all_simple_paths(
                        self, same_sub_circuit_node, neighbour_id, cutoff=self.cutoff
                    ):

                        if not all(
                            node_sub_circuit[path_node] == node_sub_circuit[node]
                            for path_node in path[:-1]
                        ):
                            same_clifford_circuit = False
                            break

                # add the neighbour if no paths in the circuit to other
                # commands in the clifford sub circuit pass through
                # non clifford sub circuits.
                if same_clifford_circuit:
                    node_sub_circuit[neighbour_id] = node_sub_circuit[node]

        if any(sub_circuit is None for sub_circuit in node_sub_circuit):  # pragma: no cover
            raise Exception("Some nodes have been left unassigned.")

        return cast(List[int], node_sub_circuit)

    # TODO: I'm not sure if this should return a circuit, or changes this
    # QermitDagCircuit in place
    def to_clifford_subcircuit_boxes(self) -> Circuit:

        # TODO: It could be worth insisting that the given circuit does not
        # include any boxes called 'Clifford Subcircuit'. i.e. that the
        # circuit is 'clean'.

        node_sub_circuit_list = self.get_clifford_subcircuits()
        sub_circuit_qubits = self.get_sub_circuit_qubits(node_sub_circuit_list)

        # List indicating if a command has been implemented
        implemented_commands = [False for _ in self.nodes()]

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
                for node_sub_circuit, implemented
                in zip(node_sub_circuit_list, implemented_commands)
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
                node for node in self.nodes()
                if node_sub_circuit_list[node] == sub_circuit_to_implement
            ]
            assert len(node_to_implement_list) > 0

            # If the circuit is clifford add it as a circbox
            if self.node_command[node_to_implement_list[0]].clifford:

                # Empty circuit to contain clifford subcircuit
                clifford_subcircuit = Circuit(
                    n_qubits=len(
                        sub_circuit_qubits[sub_circuit_to_implement]
                    ),
                    name='Clifford Subcircuit'
                )

                # Map from qubits in original circuit to qubits in new
                # clifford circuit.
                qubit_to_index = {
                    qubit: i
                    for i, qubit
                    in enumerate(sub_circuit_qubits[sub_circuit_to_implement])
                }

                # Add all gates to new circuit
                for node in node_to_implement_list:
                    clifford_subcircuit.add_gate(
                        self.node_command[node].command.op,
                        [
                            qubit_to_index[qubit]
                            for qubit in self.node_command[node].command.args
                        ]
                    )
                    implemented_commands[node] = True

                clifford_circ_box = CircBox(clifford_subcircuit)
                clifford_box_circuit.add_circbox(
                    clifford_circ_box,
                    list(sub_circuit_qubits[sub_circuit_to_implement])
                )

            # Otherwise, add the gates straight to the circuit
            else:
                for node in node_to_implement_list:
                    clifford_box_circuit.add_gate(
                        self.node_command[node].command.op,
                        self.node_command[node].command.args
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
            for qubit in self.node_command[node].command.qubits:
                sub_circuit_qubits[sub_circuit].add(qubit)

        return sub_circuit_qubits

    def can_implement(
        self,
        sub_circuit: int,
        node_sub_circuit_list: List[int],
        implemented_commands: List[bool],
    ) -> bool:

        can_implement = True
        for node in self.nodes:

            if not node_sub_circuit_list[node] == sub_circuit:
                continue

            for predecessor in self.predecessors(node):
                if node_sub_circuit_list[predecessor] == sub_circuit:
                    continue
                if not implemented_commands[predecessor]:
                    can_implement = False

        return can_implement

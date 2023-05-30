from pytket import Circuit, OpType, Qubit
import networkx as nx  # type: ignore
from pytket.circuit import Command  # type: ignore
from pytket.passes.auto_rebase import auto_rebase_pass
from .stabiliser import PauliSampler
from pytket.passes import DecomposeBoxes  # type: ignore


clifford_ops = [OpType.CZ, OpType.H, OpType.Z, OpType.S, OpType.X]
non_clifford_ops = [OpType.Rz]

cpc_rebase_pass = auto_rebase_pass(
    gateset=set(clifford_ops + non_clifford_ops)
)


class DAGCommand:

    def __init__(self, command: Command):

        self.command = command
        self.clifford = command.op.type in clifford_ops


class QermitDAGCircuit(nx.DiGraph):

    def __init__(self, circuit: Circuit):

        # TODO: There are other things to be saved, like the phase.

        super().__init__()

        self.node_command = [
            DAGCommand(command) for command in circuit.get_commands()
        ]
        self.qubits = circuit.qubits

        current_node: dict[Qubit, int] = {}
        for node, command in enumerate(self.node_command):
            self.add_node(node)
            for qubit in command.command.qubits:
                if qubit in current_node.keys():
                    self.add_edge(current_node[qubit], node)
                current_node[qubit] = node

    def get_clifford_subcircuits(self):

        # a list indicating the clifford subcircuit to which a command belongs.
        node_sub_circuit = [None for _ in self.nodes]
        sub_circuit_count = 0

        # Iterate through all commands and check if their neighbours should
        # be added to the same clifford subcircuit.
        for node, command in enumerate(self.node_command):

            # If the command is not in a clifford sub circuit, start a
            # new one and add it to that new one,
            if node_sub_circuit[node] is None:
                node_sub_circuit[node] = sub_circuit_count
                sub_circuit_count += 1

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
                # non clifford gates.
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

                # add the neighbour if no paths in the circuit to other
                # commands in the clifford sub circuit pass through
                # non clifford sub circuits.
                if same_clifford_circuit:
                    node_sub_circuit[neighbour_id] = node_sub_circuit[node]

        return node_sub_circuit

    def get_sub_circuit_qubits(
        self, node_sub_circuit: list[int]
    ) -> dict[int, set[Qubit]]:
        """Creates a dictionary from the clifford sub circuit to the qubits
        which it covers

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
        sub_circuit,
        node_sub_circuit_list,
        implemented_commands,
    ):

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

    def add_pauli_checks(self, pauli_sampler: PauliSampler):

        node_sub_circuit_list = self.get_clifford_subcircuits()
        sub_circuit_qubits = self.get_sub_circuit_qubits(node_sub_circuit_list)

        # List indicating if a command has been implemented
        implemented_commands = [False for _ in self.nodes()]

        # Initialise new circuit
        pauli_check_circuit = Circuit()
        for quibt in self.qubits:
            pauli_check_circuit.add_qubit(quibt)

        ancilla_count = 0

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

            qubit_list = list(sub_circuit_qubits[sub_circuit_to_implement])

            # Add a barrier is this is a Clifford sub circuit
            if self.node_command[node_to_implement_list[0]].clifford:

                control_qubit = Qubit(
                    name='ancilla',
                    index=ancilla_count,
                )
                pauli_check_circuit.add_qubit(control_qubit)
                ancilla_count += 1

                pauli_check_circuit.add_barrier(
                    qubit_list
                )

                stabiliser = pauli_sampler.sample(qubit_list=qubit_list)
                stabiliser_circuit = stabiliser.get_control_circuit(
                    control_qubit=control_qubit
                )
                pauli_check_circuit.append(
                    circuit=stabiliser_circuit,
                )

                pauli_check_circuit.add_barrier(
                    qubit_list
                )

            # Add all commands in the sub circuit
            for node in node_to_implement_list:
                pauli_check_circuit.add_gate(
                    self.node_command[node].command.op,
                    self.node_command[node].command.qubits
                )
                implemented_commands[node] = True
                if self.node_command[node].clifford:
                    stabiliser.apply_gate(
                        self.node_command[node].command.op.type,
                        self.node_command[node].command.qubits
                    )

            # Add barrier if this is a Clifford sub circuit.
            if self.node_command[node_to_implement_list[0]].clifford:

                pauli_check_circuit.add_barrier(
                    qubit_list
                )

                stabiliser_circuit = stabiliser.get_control_circuit(
                    control_qubit=control_qubit
                )
                pauli_check_circuit.append(
                    circuit=stabiliser_circuit,
                )

                pauli_check_circuit.add_barrier(
                    qubit_list
                )

        DecomposeBoxes().apply(pauli_check_circuit)

        # TODO: Given more time it would be nice to add a check here
        # which XZ reduces the circuits with and without the checks and
        # asserts that they are the same.

        return pauli_check_circuit

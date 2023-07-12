from pytket import Circuit, OpType, Qubit
import networkx as nx  # type: ignore
from pytket.circuit import Command, Bit  # type: ignore
from pytket.passes.auto_rebase import auto_rebase_pass
from .pauli_sampler import PauliSampler
from pytket.passes import DecomposeBoxes  # type: ignore
from pytket.circuit import CircBox


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
        self.bits = circuit.bits

        # Lists the most recent node to act on a particular qubits. If a
        # new gate is found to act on that qubit then an edge between the
        # node which corresponds to the new gate and the node which
        # most recently acted on that qubit is added.
        current_node: dict[Qubit, int] = {}
        for node, command in enumerate(self.node_command):
            self.add_node(node)
            for qubit in command.command.qubits:
                # This if statement is used in case the qubit has not been
                # acted on yet.
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
                # a different Clifford subcircuit. If nodes on the path
                # belonged to another clifford subcircuit then it would
                # not be possible to build the circuit by applying
                # sub circuits sequentially.
                same_clifford_circuit = True
                for same_sub_circuit_node in same_sub_circuit_node_list:

                    for path in nx.all_simple_paths(
                        self, same_sub_circuit_node, neighbour_id
                    ):

                        if not all(
                            node_sub_circuit[path_node] == node_sub_circuit[node]
                            for path_node in path[:-1]
                        ):
                            same_clifford_circuit = False

                # add the neighbour if no paths in the circuit to other
                # commands in the clifford sub circuit pass through
                # non clifford sub circuits.
                if same_clifford_circuit:
                    node_sub_circuit[neighbour_id] = node_sub_circuit[node]

        return node_sub_circuit

    def to_clifford_subcircuit_boxes(self):

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
                    n_qubits = len(
                        sub_circuit_qubits[sub_circuit_to_implement]
                    ),
                    name = 'Clifford Subcircuit'
                )
                
                # Map from qubits in original circuit to qubits in new
                # clifford circuit.
                qubit_to_index = {
                    qubit:i
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

    def add_pauli_checks_to_circbox(self, pauli_sampler: PauliSampler):

        pauli_check_circuit = Circuit()
        for qubit in self.qubits:
            pauli_check_circuit.add_qubit(qubit)
        for bit in self.bits:
            pauli_check_circuit.add_bit(bit)
            
        ancilla_count = 0

        # Add each command in the circuit, wrapped by checks
        # if the command is a circbox named 'Clifford Subcircuit'
        for command in [
            node_command.command for node_command in self.node_command
        ]:
            
            # Add barriers and check if appropriate
            if (
                command.op.type == OpType.CircBox
            ) and (
                command.op.get_circuit().name == 'Clifford Subcircuit'
            ):
                
                clifford_subcircuit = command.op.get_circuit()
                
                control_qubit = Qubit(
                    name='ancilla',
                    index=ancilla_count,
                )
                # TODO: check that register names do not already exist
                pauli_check_circuit.add_qubit(control_qubit)

                pauli_check_circuit.add_barrier(
                    command.args + [control_qubit]
                )
                pauli_check_circuit.H(control_qubit)

                stabiliser = pauli_sampler.sample(qubit_list=command.args)
                stabiliser_circuit = stabiliser.get_control_circuit(
                    control_qubit=control_qubit
                )
                pauli_check_circuit.append(
                    circuit=stabiliser_circuit,
                )

                pauli_check_circuit.add_barrier(
                    command.args + [control_qubit]
                )
                
            # Add command
            pauli_check_circuit.add_gate(
                command.op,
                command.args
            )
            
            # Add barriers and checks if appropriate.
            if (
                command.op.type == OpType.CircBox
            ) and (
                command.op.get_circuit().name == 'Clifford Subcircuit'
            ):
                
                qubit_map = {
                    q_subcirc: q_orig
                    for q_subcirc, q_orig
                    in zip(clifford_subcircuit.qubits, command.args)
                }
                for clifford_command in clifford_subcircuit.get_commands():
                    # TODO: an error would be raised here if clifford_command
                    # is not Clifford. It could be worth raising a clearer
                    # error.
                    stabiliser.apply_gate(
                        clifford_command.op.type,
                        [qubit_map[qubit] for qubit in clifford_command.qubits]
                    )
                    
                pauli_check_circuit.add_barrier(
                    command.args + [control_qubit]
                )

                stabiliser_circuit = stabiliser.get_control_circuit(
                    control_qubit=control_qubit
                )
                pauli_check_circuit.append(
                    circuit=stabiliser_circuit,
                )
                pauli_check_circuit.H(control_qubit)
                pauli_check_circuit.add_barrier(
                    command.args + [control_qubit]
                )

                measure_bit = Bit(
                    name='ancilla_measure',
                    index=ancilla_count,
                )
                pauli_check_circuit.add_bit(
                    id=measure_bit
                )
                pauli_check_circuit.Measure(
                    qubit=control_qubit,
                    bit=measure_bit,
                )

                ancilla_count += 1
                
        return pauli_check_circuit

    def add_pauli_checks(self, pauli_sampler: PauliSampler):

        # Convert to clifford boxes, add checks, decompose boxes.
        clifford_box_circuit = self.to_clifford_subcircuit_boxes()
        cliff_box_dag_circ = QermitDAGCircuit(clifford_box_circuit)
        pauli_check_circ = cliff_box_dag_circ.add_pauli_checks_to_circbox(
            pauli_sampler=pauli_sampler
        )
        DecomposeBoxes().apply(pauli_check_circ)

        # TODO: Given more time it would be nice to add a check here
        # which XZ reduces the circuits with and without the checks and
        # asserts that they are the same.

        return pauli_check_circ

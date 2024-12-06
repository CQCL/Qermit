import networkx as nx  # type: ignore
from pytket import Circuit, Qubit
from pytket.circuit import CircBox, Circuit, Command, OpType
from pytket.passes import BasePass, CustomPass

from .monochromatic_convex_subdag import MonochromaticConvexSubDAG


def command_is_clifford(command: Command) -> bool:
    """Check if the given command is clifford.

    :param command: Command to check.
    :return: Boolean value indicating if given command is clifford.
    """

    # This is only a limited set of gates. This should be expanded.

    if command.op.is_clifford_type():
        return True

    if command.op.type == OpType.Rz:
        if command.op.params == [0.5]:
            return True

    if command.op.type == OpType.PhasedX:
        if command.op.params == [0.5, 0.5]:
            return True

    return False


def get_clifford_commands(command_list: list[Command]) -> set[int]:
    """Given a list of command, return a set of indexes of that list
    corresponding to those commands which are Clifford gates.

    :param command_list: List of commands in which to search for
        Clifford commends.
    :return: Indexes in the list which correspond to commands in the
        list which are Clifford.
    """
    return {i for i, command in enumerate(command_list) if command_is_clifford(command)}


def get_clifford_subcircuits(dag, clifford_nodes) -> list[int]:
    node_subdag = MonochromaticConvexSubDAG(
        dag=dag,
        coloured_nodes=clifford_nodes,
    ).greedy_merge()
    node_sub_circuit_list = []
    sub_circuit_index = 0
    for node in range(dag.number_of_nodes()):
        if node in node_subdag.keys():
            node_sub_circuit_list.append(node_subdag[node])
        else:
            while (sub_circuit_index in node_subdag.values()) or (
                sub_circuit_index in node_sub_circuit_list
            ):
                sub_circuit_index += 1
            node_sub_circuit_list.append(sub_circuit_index)

    return node_sub_circuit_list


def circuit_to_graph(circuit: Circuit) -> tuple[nx.DiGraph, list[Command]]:
    """Convert circuit to graph. Nodes correspond to commands,
    edges indicate a dependence between the outputs and inputs of
    two commands. Node values corresponds to indexes in the returned
    list of commands.

    :param circuit: Circuit to convert to a graph.
    :return: Tuple of graph and list of commands. Nodes are indexes
        in the list of commands.
    """
    # Lists the most recent node to act on a particular qubits. If a
    # new gate is found to act on that qubit then an edge between the
    # node which corresponds to the new gate and the node which
    # most recently acted on that qubit is added. This builds a DAG
    # showing the temporal order of gates.
    node_command = circuit.get_commands()
    current_node: dict[Qubit, int] = {}
    dag = nx.DiGraph()
    for node, command in enumerate(node_command):
        dag.add_node(node)
        for qubit in command.qubits:
            # This if statement is used in case the qubit has not been
            # acted on yet.
            if qubit in current_node.keys():
                dag.add_edge(current_node[qubit], node)
            current_node[qubit] = node

    return dag, node_command


def get_sub_circuit_qubits(
    command_list: list[Command],
    command_subcircuit: list[int],
) -> dict[int, set[Qubit]]:
    """For each subcircuit, get the qubits on which it acts.

    :param command_list: A list of commands.
    :param command_subcircuit: The subcircuit to which each command belongs.
    :return: A map from the subcircuit to the qubits it act on.
    """
    sub_circuit_list = list(set(command_subcircuit))
    sub_circuit_qubits: dict[int, set[Qubit]] = {
        sub_circuit: set() for sub_circuit in sub_circuit_list
    }
    for node, sub_circuit in enumerate(command_subcircuit):
        for qubit in command_list[node].qubits:
            sub_circuit_qubits[sub_circuit].add(qubit)

    return sub_circuit_qubits


def can_implement(
    sub_circuit: int,
    node_sub_circuit_list: list[int],
    implemented_commands: list[bool],
    dag,
    node_command,
) -> bool:
    can_implement = True
    for node in range(len(node_command)):
        if not node_sub_circuit_list[node] == sub_circuit:
            continue

        for predecessor in dag.predecessors(node):
            if node_sub_circuit_list[predecessor] == sub_circuit:
                continue
            if not implemented_commands[predecessor]:
                can_implement = False

    return can_implement


def box_clifford_transform(circuit: Circuit) -> Circuit:
    dag, node_command = circuit_to_graph(circuit=circuit)
    clifford_nodes = get_clifford_commands(node_command)
    node_sub_circuit_list = get_clifford_subcircuits(
        dag=dag, clifford_nodes=clifford_nodes
    )
    sub_circuit_qubits = get_sub_circuit_qubits(
        command_list=node_command,
        command_subcircuit=node_sub_circuit_list,
    )

    # List indicating if a command has been implemented
    implemented_commands = [False] * len(node_command)

    # Initialise new circuit
    clifford_box_circuit = Circuit()
    for qubit in circuit.qubits:
        clifford_box_circuit.add_qubit(qubit)
    for bit in circuit.bits:
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
            if can_implement(
                sub_circuit,
                node_sub_circuit_list,
                implemented_commands,
                dag,
                node_command,
            ):
                sub_circuit_to_implement = sub_circuit
                break
        assert sub_circuit_to_implement is not None

        # List the nodes in the chosen sub circuit
        node_to_implement_list = [
            node
            for node in range(len(node_command))
            if node_sub_circuit_list[node] == sub_circuit_to_implement
        ]
        assert len(node_to_implement_list) > 0

        # If the circuit is clifford add it as a circbox
        if node_to_implement_list[0] in clifford_nodes:
            # Empty circuit to contain clifford subcircuit
            clifford_subcircuit = Circuit(
                n_qubits=len(sub_circuit_qubits[sub_circuit_to_implement]),
                name="Clifford Subcircuit",
            )

            # Map from qubits in original circuit to qubits in new
            # clifford circuit.
            qubit_to_index = {
                qubit: i
                for i, qubit in enumerate(sub_circuit_qubits[sub_circuit_to_implement])
            }

            # Add all gates to new circuit
            for node in node_to_implement_list:
                clifford_subcircuit.add_gate(
                    node_command[node].op,
                    [qubit_to_index[qubit] for qubit in node_command[node].args],
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
                    node_command[node].op,
                    node_command[node].args,
                )
                implemented_commands[node] = True

    return clifford_box_circuit


def BoxClifford() -> BasePass:
    return CustomPass(transform=box_clifford_transform)

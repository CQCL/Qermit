import networkx as nx  # type: ignore
from pytket._tket.unit_id import Qubit
from pytket.circuit import CircBox, Circuit, Command, OpType
from pytket.passes import BasePass, CustomPass

from .monochromatic_convex_subdag import get_monochromatic_convex_subdag


def _command_is_clifford(command: Command) -> bool:
    """Check if the given command is Clifford.

    :param command: Command to check.
    :return: Boolean value indicating if given command is Clifford.
    """

    # This is only a limited set of gates.
    # TODO: This should be expanded.

    if command.op.is_clifford_type():
        return True

    if command.op.type == OpType.Rz:
        if command.op.params == [0.5]:
            return True

    if command.op.type == OpType.PhasedX:
        if command.op.params == [0.5, 0.5]:
            return True

    return False


def _get_clifford_commands(command_list: list[Command]) -> list[int]:
    """Given a list of commands, return a set of indexes of that list
    corresponding to those commands which are Clifford gates.

    :param command_list: List of commands in which to search for
        Clifford commends.
    :return: Indexes in the list which correspond to commands in the
        list which are Clifford.
    """
    return [
        i for i, command in enumerate(command_list) if _command_is_clifford(command)
    ]


def _give_nodes_subdag(dag: nx.DiGraph, node_subdag: dict[int, int]) -> list[int]:
    """Assign a sub-DAG to all nodes in given dag. Some may already have
    an assigned sub-DAG as given and these are preserved. Nodes without an
    assigned sub-DAG are given a unique sub-DAG of their own.

    :param dag: Directed acyclic graph.
    :param node_subdag: Map from node to sub-DAG for those nodes with
        an existing assignment.
    :raises Exception: Raised if nodes are not sequential integers.
    :return: List of sub-DAGs. List is indexed by node.
    """

    if not sorted(list(dag.nodes)) == [i for i in range(dag.number_of_nodes())]:
        raise Exception("The nodes of the given dag must be sequential integers.")

    node_subdag_list = []
    subdag_index = 0
    for node in range(dag.number_of_nodes()):
        if node in node_subdag.keys():
            node_subdag_list.append(node_subdag[node])
        else:
            while (subdag_index in node_subdag.values()) or (
                subdag_index in node_subdag_list
            ):
                subdag_index += 1
            node_subdag_list.append(subdag_index)

    return node_subdag_list


def _circuit_to_graph(circuit: Circuit) -> tuple[nx.DiGraph, list[Command]]:
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


def _get_sub_circuit_qubits(
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


def _can_implement(
    sub_circuit: int,
    command_sub_circuit: list[int],
    command_implemented: list[bool],
    dag: nx.DiGraph,
    node_command: list[Command],
) -> bool:
    """True if it is safe to implement a subcircuit. False otherwise.
    This will be true if all predecessors of commands in the sub circuit
    have been implemented.

    :param sub_circuit: Subcircuit to check.
    :param command_sub_circuit: The subcircuit of each command.
    :param command_implemented: List with entry for each command indicating if
        it has been implemented.
    :param dag: Graph giving dependencies between commands.
    :param node_command: Command corresponding to each node in the graph.
    :return: True if it is safe to implement a subcircuit. False otherwise.
    """
    _can_implement = True
    for node in range(len(node_command)):
        if not command_sub_circuit[node] == sub_circuit:
            continue

        for predecessor in dag.predecessors(node):
            if command_sub_circuit[predecessor] == sub_circuit:
                continue
            if not command_implemented[predecessor]:
                _can_implement = False

    return _can_implement


def _box_clifford_transform(circuit: Circuit) -> Circuit:
    """Replace Clifford subcircuits with boxes containing those circuits.
    These boxes will have the name "Clifford Subcircuit".

    :param circuit: Circuit whose Clifford subcircuits should be boxed.
    :return: Equivalent circuit with subcircuits boxed.
    :rtype: Circuit
    """
    dag, node_command = _circuit_to_graph(circuit=circuit)
    clifford_nodes = _get_clifford_commands(node_command)

    node_subdag = get_monochromatic_convex_subdag(
        dag=dag,
        coloured_nodes=clifford_nodes,
    )

    node_sub_circuit_list = _give_nodes_subdag(dag=dag, node_subdag=node_subdag)

    sub_circuit_qubits = _get_sub_circuit_qubits(
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
    clifford_box_circuit.add_phase(circuit.phase)

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
            if _can_implement(
                sub_circuit=sub_circuit,
                command_sub_circuit=node_sub_circuit_list,
                command_implemented=implemented_commands,
                dag=dag,
                node_command=node_command,
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
            assert all(
                node_to_implement in clifford_nodes
                for node_to_implement in node_to_implement_list
            )

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
                # It is assumed that the commands have no classical bits.
                if node_command[node].args != node_command[node].qubits:
                    raise Exception(
                        "This Clifford subcircuit contains classical bits."
                        "This is a bug and should be reported to the developers."
                    )

                if node_command[node].op.type == OpType.Barrier:
                    raise Exception(
                        "This Clifford subcircuit contains a barrier."
                        "This is a bug and should be reported to the developers."
                    )

                if node_command[node].opgroup is not None:
                    clifford_subcircuit.add_gate(
                        Op=node_command[node].op,
                        args=[
                            qubit_to_index[qubit] for qubit in node_command[node].qubits
                        ],
                        opgroup=node_command[node].opgroup,
                    )

                else:
                    clifford_subcircuit.add_gate(
                        Op=node_command[node].op,
                        args=[
                            qubit_to_index[qubit] for qubit in node_command[node].qubits
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
            assert len(node_to_implement_list) == 1

            if node_command[node_to_implement_list[0]].op.type == OpType.Barrier:
                clifford_box_circuit.add_barrier(
                    units=node_command[node_to_implement_list[0]].args,
                    data=node_command[node_to_implement_list[0]].op.data,  # type: ignore
                )

            else:
                clifford_box_circuit.add_gate(
                    node_command[node_to_implement_list[0]].op,
                    node_command[node_to_implement_list[0]].args,
                )

            implemented_commands[node_to_implement_list[0]] = True

    return clifford_box_circuit


def BoxClifford() -> BasePass:
    """
    Pass finding clifford subcircuits and wrapping them
    in circuit boxed called "Clifford Subcircuit".

    :return: Pass finding clifford subcircuits and wrapping them
        in circuit boxed called "Clifford Subcircuit".
    """
    return CustomPass(transform=_box_clifford_transform)

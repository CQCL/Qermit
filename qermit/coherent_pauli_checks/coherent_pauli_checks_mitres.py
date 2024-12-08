from typing import List, Tuple

from pytket.backends import Backend
from pytket.passes import DecomposeBoxes

from qermit import CircuitShots, MitRes, MitTask, TaskGraph
from qermit.coherent_pauli_checks.box_clifford_subcircuits import BoxClifford
from qermit.postselection.postselect_manager import PostselectMgr
from qermit.postselection.postselect_mitres import gen_postselect_task

from .pauli_sampler import PauliSampler


def gen_find_cliffords_task() -> MitTask:
    """Generator for MitTask for finding Clifford subcircuits within
    given circuits. Inputs and outputs are CircuitShots. Output circuits
    have Clifford sub circuits wrapped in CircBox.

    :return: Task for finding Clifford subcircuits.
    """

    def task(_, circ_shots_list: List[CircuitShots]) -> Tuple[List[CircuitShots]]:
        """Function wrapping Clifford subcircuits in CircBox.

        :param circ_shots_list: List of circuit shots.
        :return: Identical circuits with Clifford circuit wrapped
            in CircBox.
        """
        cliff_circ_shots_list = []

        for circ_shots in circ_shots_list:
            cliff_circ = circ_shots.Circuit.copy()
            BoxClifford().apply(cliff_circ)
            cliff_circ_shots_list.append(
                CircuitShots(
                    Circuit=cliff_circ,
                    Shots=circ_shots.Shots,
                )
            )

        return (cliff_circ_shots_list,)

    return MitTask(
        _label="FindCliffordSubcircuits",
        _n_in_wires=1,
        _n_out_wires=1,
        _method=task,
    )


def gen_check_circuit_task(pauli_sampler: PauliSampler) -> MitTask:
    """Generator for MitTask applying Pauli checks to circuits.

    :param pauli_sampler: PauliSampler to use to sample checks.
    :return: Task for adding Pauli checks.
    """

    def task(
        _, circ_shots_list: List[CircuitShots]
    ) -> Tuple[List[CircuitShots], List[PostselectMgr]]:
        """Task for adding Pauli checks to given circuits.

        :param circ_shots_list: Circuits to add Pauli checks to.
        :return: Circuits with pauli checks added.
        """
        checked_circ_shots_list = []
        postselect_mgr_list = []

        for circ_shots in circ_shots_list:
            checked_circuit, postselect_cbits = (
                pauli_sampler.add_pauli_checks_to_circbox(circuit=circ_shots.Circuit)
            )

            postselect_mgr_list.append(
                PostselectMgr(
                    compute_cbits=circ_shots.Circuit.bits,
                    postselect_cbits=list(postselect_cbits),
                )
            )

            DecomposeBoxes().apply(checked_circuit)

            checked_circ_shots_list.append(
                CircuitShots(
                    Circuit=checked_circuit,
                    Shots=circ_shots.Shots,
                )
            )

        return (checked_circ_shots_list, postselect_mgr_list)

    return MitTask(
        _label="CheckCircuits",
        _n_in_wires=1,
        _n_out_wires=2,
        _method=task,
    )


def gen_coherent_pauli_check_mitres(
    backend: Backend,
    pauli_sampler: PauliSampler,
) -> MitRes:
    """Generator for MitRes performing mitigation through
    Coherent Pauli Checks.

    :param backend: Backend to perform perform experiment with.
    :param pauli_sampler: Sampler to use to apply Pauli checks.
    :return: MitRes performing mitigation through
    Coherent Pauli Checks.
    """
    _mitres = MitRes(backend, _label="PostselectionMitRes")
    _taskgraph = TaskGraph().from_TaskGraph(_mitres)

    _taskgraph.add_wire()

    _taskgraph.prepend(gen_check_circuit_task(pauli_sampler))
    _taskgraph.prepend(gen_find_cliffords_task())
    _taskgraph.append(gen_postselect_task())

    return MitRes(backend).from_TaskGraph(_taskgraph)

from copy import deepcopy
from typing import List, Tuple

from pytket.backends import Backend
from pytket.backends.backendresult import BackendResult

from qermit import CircuitShots, MitRes, MitTask, TaskGraph

from .postselect_manager import PostselectMgr


def gen_postselect_task() -> MitTask:
    """Generates task applying postselection to given results.

    :return: Task applying postselection to given results.
    """

    def task(
        obj,
        result_list: List[BackendResult],
        postselect_mgr_list: List[PostselectMgr],
    ) -> Tuple[List[BackendResult]]:
        """Task applying postselection to given results.

        :param result_list: List od results to which postselection should
            be applied.
        :param postselect_mgr_list: List of postselection managers to apply
            to results.
        :return: List of results after postselection has been applied.
        """

        return (
            [
                postselect_mgr.postselect_result(result)
                for result, postselect_mgr in zip(result_list, postselect_mgr_list)
            ],
        )

    return MitTask(
        _label="PostselectResults",
        _n_in_wires=2,
        _n_out_wires=1,
        _method=task,
    )


def gen_postselect_mgr_gen_task(postselect_mgr: PostselectMgr) -> MitTask:
    """Generates task applying the same post selection manager to all
    circuits.

    :param postselect_mgr: Postselection manager to apply.
    :return: Task applying the same post selection manager to all circuits.
    """

    def task(
        obj, circ_shots_list: List[CircuitShots]
    ) -> Tuple[List[CircuitShots], List[PostselectMgr]]:
        """Task applying the same post selection manager to all circuits.

        :param circ_shots_list: List of circuits to which post selection is
            applied.
        :return: List od circuits and corresponding postselection managers.
        """

        return (circ_shots_list, [postselect_mgr for _ in circ_shots_list])

    return MitTask(
        _label="UniformPostselection",
        _n_in_wires=1,
        _n_out_wires=2,
        _method=task,
    )


def gen_postselect_mitres(
    backend: Backend, postselect_mgr: PostselectMgr, **kwargs
) -> MitRes:
    """Generates MitRes running given circuit and applying postselection.

    In the following example we prepare and measure a Bell state.

    .. jupyter-execute::

        from pytket import Circuit
        from pytket.circuit.display import render_circuit_jupyter

        circuit = Circuit(2,2).H(0).CX(0,1).measure_all()
        render_circuit_jupyter(circuit)

    We would like to postselect one measurement outcome based on the
    other being 0. We prepare a postselect mitres accordingly.

    .. jupyter-execute::

        from qermit.postselection import gen_postselect_mitres
        from pytket.extensions.quantinuum import QuantinuumBackend, QuantinuumAPIOffline
        from qermit.postselection import PostselectMgr

        backend = QuantinuumBackend(
            device_name="H1-1LE",
            api_handler = QuantinuumAPIOffline()
        )

        postselect_mgr = PostselectMgr(
            compute_cbits=[circuit.bits[0]],
            postselect_cbits=[circuit.bits[1]]
        )

        mitres = gen_postselect_mitres(
            backend=backend,
            postselect_mgr=postselect_mgr,
        )
        mitres.get_task_graph()

    We can then construct the experiment we wish to run, and pass it through
    the postselect mitres.

    .. jupyter-execute::

        from qermit import CircuitShots

        circ_shots = CircuitShots(
            Circuit=backend.get_compiled_circuit(circuit),
            Shots=100
        )
        result_list = mitres.run([circ_shots])
        result_list[0].get_counts()

    :param backend: Backend on this circuits are run.
    :param postselect_mgr: Postselection manager.
    :return: MitRes running given circuit and applying postselection.
    """

    _mitres = deepcopy(
        kwargs.get("mitres", MitRes(backend, _label="PostselectionMitRes"))
    )
    _taskgraph = TaskGraph().from_TaskGraph(_mitres)
    _taskgraph.add_wire()
    _taskgraph.prepend(gen_postselect_mgr_gen_task(postselect_mgr))
    _taskgraph.append(gen_postselect_task())

    return MitRes(backend).from_TaskGraph(_taskgraph)

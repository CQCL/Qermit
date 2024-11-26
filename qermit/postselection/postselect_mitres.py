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
        _label="ConstantNode",
        _n_in_wires=1,
        _n_out_wires=2,
        _method=task,
    )


def gen_postselect_mitres(
    backend: Backend, postselect_mgr: PostselectMgr, **kwargs
) -> MitRes:
    """Generates MitRes running given circuit and applying postselection.

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

from .postselect_manager import PostselectMgr
from qermit import CircuitShots, MitRes, MitTask, TaskGraph
from copy import deepcopy
from typing import List, Tuple
from pytket.backends.backendresult import BackendResult
from pytket.backends import Backend


def gen_postselect_task() -> MitTask:
    """Generates task applying postselection to given results.

    :return: Task applying postselection to given results.
    :rtype: MitTask
    """

    def task(
        obj,
        result_list: List[BackendResult],
        postselect_mgr_list: List[PostselectMgr],
    ) -> Tuple[List[BackendResult]]:
        """Task applying postselection to given results.

        :param result_list: List od results to which postselection should
            be applied.
        :type result_list: List[BackendResult]
        :param postselect_mgr_list: List of postselection managers to apply
            to results.
        :type postselect_mgr_list: List[PostselectMgr]
        :return: List of results after postselection has been applied.
        :rtype: Tuple[List[BackendResult]]
        """

        return (
            [
                postselect_mgr.post_select_result(result)
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
    :type postselect_mgr: PostselectMgr
    :return: Task applying the same post selection manager to all circuits.
    :rtype: MitTask
    """

    def task(
        obj,
        circ_shots_list: List[CircuitShots]
    ) -> Tuple[List[CircuitShots], List[PostselectMgr]]:
        """Task applying the same post selection manager to all circuits.

        :param circ_shots_list: List of circuits to which post selection is
            applied.
        :type circ_shots_list: List[CircuitShots]
        :return: List od circuits and corresponding postselection managers.
        :rtype: Tuple[List[CircuitShots], List[PostselectMgr]]
        """

        return (circ_shots_list, [postselect_mgr for _ in circ_shots_list])

    return MitTask(
        _label="ConstantNode",
        _n_in_wires=1,
        _n_out_wires=2,
        _method=task,
    )


def gen_postselect_mitres(
    backend: Backend,
    postselect_mgr: PostselectMgr,
    **kwargs
) -> MitRes:
    """Generates MitRes running given circuit and applying postselection.

    :param backend: Backend on this circuits are run.
    :type backend: Backend
    :param postselect_mgr: Postselection manager.
    :type postselect_mgr: PostselectMgr
    :return: MitRes running given circuit and applying postselection.
    :rtype: MitRes
    """

    _mitres = deepcopy(
        kwargs.get("mitres", MitRes(backend, _label="PostselectionMitRes"))
    )
    _taskgraph = TaskGraph().from_TaskGraph(_mitres)
    _taskgraph.add_wire()
    _taskgraph.prepend(gen_postselect_mgr_gen_task(postselect_mgr))
    _taskgraph.append(gen_postselect_task())

    return MitRes(backend).from_TaskGraph(_taskgraph)

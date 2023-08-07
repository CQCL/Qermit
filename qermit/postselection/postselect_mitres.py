from pytket.extensions.qiskit import AerBackend
from pytket import Circuit
from .postselect_manager import PostselectMgr
from qermit import CircuitShots, MitRes, MitTask, TaskGraph
from copy import deepcopy
from typing import List, Tuple
from pytket.backends.backendresult import BackendResult


def gen_postselect_task():
    
    def task(
        obj,
        result_list:List[BackendResult],
        postselect_mgr_list:[PostselectMgr],
    ) -> Tuple[List[BackendResult]]:
                        
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


def gen_postselect_mgr_gen_task(postselect_mgr):
    
    def task(obj, circ_shots_list):
        
        return (circ_shots_list, [postselect_mgr for _ in circ_shots_list])
    
    return MitTask(
        _label="ConstantNode",
        _n_in_wires=1,
        _n_out_wires=2,
        _method=task,
    )


def gen_postselect_mitres(backend, postselect_mgr, **kwargs):
    
    _mitres = deepcopy(
        kwargs.get("mitres", MitRes(backend, _label="PostselectionMitRes"))
    )
    _taskgraph = TaskGraph().from_TaskGraph(_mitres)
    _taskgraph.add_wire()
    _taskgraph.prepend(gen_postselect_mgr_gen_task(postselect_mgr))
    _taskgraph.append(gen_postselect_task())
    
    return MitRes(backend).from_TaskGraph(_taskgraph)
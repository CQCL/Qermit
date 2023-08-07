from pytket.extensions.qiskit import AerBackend
from pytket import Circuit
from qermit.postselection import PostselectMgr, gen_postselect_mitres
from qermit.postselection.postselect_mitres import gen_postselect_task
from qermit import CircuitShots, MitRes, MitTask, TaskGraph
from copy import deepcopy
from typing import List, Tuple
from pytket.backends.backendresult import BackendResult
from pytket.extensions.quantinuum.backends.leakage_gadget import get_detection_circuit
from qermit.mock_backend import MockQuantinuumBackend
from qermit.taskgraph import gen_compiled_MitRes

def gen_add_leakage_gadget_circuit_task(backend):
    
    def task(
        obj,
        circuit_shots_list:List[CircuitShots]
    ) -> Tuple[List[CircuitShots], List[PostselectMgr]]:
        
        detection_circuit_shots_list = [
            CircuitShots(
                Circuit = get_detection_circuit(circuit_shots.Circuit, backend.backend_info.n_nodes),
                Shots = circuit_shots.Shots,
            )
            for circuit_shots in circuit_shots_list
        ]
        
        postselect_mgr_list = [
            PostselectMgr(
                compute_cbits=orig_circuit.Circuit.bits,
                postselect_cbits=list(
                    set(detection_circuit.Circuit.bits).difference(set(orig_circuit.Circuit.bits))
                )
            )
            for orig_circuit, detection_circuit in zip(circuit_shots_list, detection_circuit_shots_list)
        ]
        
        return (detection_circuit_shots_list, postselect_mgr_list, )
                        
    return MitTask(
        _label="AddLeakageGadget",
        _n_in_wires=1,
        _n_out_wires=2,
        _method=task,
    )

def get_leakage_gadget_mitres(backend, **kwargs):
    
    _mitres = deepcopy(
        kwargs.get("mitres", MitRes(backend, _label="LeakageGadgetMitRes"))
    )
    _taskgraph = TaskGraph().from_TaskGraph(_mitres)
    _taskgraph.add_wire()
    _taskgraph.prepend(gen_add_leakage_gadget_circuit_task(backend))
    _taskgraph.append(gen_postselect_task())
    
    return MitRes(backend).from_TaskGraph(_taskgraph)
from copy import deepcopy
from typing import List, Tuple, cast

from pytket.backends import Backend
from pytket.backends.backendinfo import BackendInfo
from pytket.extensions.quantinuum.backends.leakage_gadget import get_detection_circuit

from qermit import CircuitShots, MitRes, MitTask, TaskGraph
from qermit.postselection import PostselectMgr
from qermit.postselection.postselect_mitres import gen_postselect_task


def gen_add_leakage_gadget_circuit_task(backend: Backend) -> MitTask:
    """Generates task adding leakage gadget circuits to given circuts.

    :param backend: Backend on which the circuit will be run.
    :return: Task adding leakage gadget circuits to given circuts.
    """

    if backend.backend_info is None:
        raise Exception("This backend has no nodes.")

    n_device_qubits = cast(BackendInfo, backend.backend_info).n_nodes

    def task(
        obj, circuit_shots_list: List[CircuitShots]
    ) -> Tuple[List[CircuitShots], List[PostselectMgr]]:
        """Task adding leakage gadget circuits to given circuts. This reuses
        methods from pytket-quantinuum. A list of the corresponding
        postselection managers is also created.

        :param circuit_shots_list: List of circuits to which leakage gadget
            circuit should be added.
        :return: Circuits with gadget added, and list of corresponding
            post selection managers.
        """

        # Add leakage detection gadget to each inputted circuit.
        detection_circuit_shots_list = [
            CircuitShots(
                Circuit=get_detection_circuit(
                    circuit=circuit_shots.Circuit,
                    n_device_qubits=n_device_qubits,
                ),
                Shots=circuit_shots.Shots,
            )
            for circuit_shots in circuit_shots_list
        ]

        # For each circuit create a postselection manager. These may be
        # different for each circuit, if for example the circuits are of
        # different sizes.
        postselect_mgr_list = [
            PostselectMgr(
                compute_cbits=orig_circuit.Circuit.bits,
                postselect_cbits=list(
                    set(detection_circuit.Circuit.bits).difference(
                        set(orig_circuit.Circuit.bits)
                    )
                ),
            )
            for orig_circuit, detection_circuit in zip(
                circuit_shots_list, detection_circuit_shots_list
            )
        ]

        return (
            detection_circuit_shots_list,
            postselect_mgr_list,
        )

    return MitTask(
        _label="AddLeakageGadget",
        _n_in_wires=1,
        _n_out_wires=2,
        _method=task,
    )


def get_leakage_detection_mitres(backend: Backend, **kwargs) -> MitRes:
    """Generate MitRes making use of leakage detection and postselection.

    :param backend: Backend on which the circuits are run.
    :return: MitRes making use of leakage detection and postselection.
    """

    _mitres = deepcopy(
        kwargs.get("mitres", MitRes(backend, _label="LeakageDetectionMitRes"))
    )
    _taskgraph = TaskGraph().from_TaskGraph(_mitres)
    _taskgraph.add_wire()
    # Prepend task adding leakage detection circuits.
    _taskgraph.prepend(gen_add_leakage_gadget_circuit_task(backend))
    # Append task removing shots where leakage is detected
    _taskgraph.append(gen_postselect_task())

    return MitRes(backend).from_TaskGraph(_taskgraph)

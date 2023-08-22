from qermit.mock_backend import MockQuantinuumBackend
from pytket import Circuit
from qermit.taskgraph import gen_compiled_MitRes
from qermit import CircuitShots
from qermit.leakage_detection import get_leakage_detection_mitres
from qermit.leakage_detection.leakage_detection import gen_add_leakage_gadget_circuit_task
from pytket.extensions.quantinuum.backends.leakage_gadget import prune_shots_detected_as_leaky
from qermit.postselection.postselect_mitres import gen_postselect_task


def test_leakage_gadget() -> None:

    backend = MockQuantinuumBackend()
    circuit = Circuit(2).H(0).measure_all()
    compiled_mitres = gen_compiled_MitRes(
        backend=backend,
        optimisation_level=0,
    )
    leakage_gadget_mitres = get_leakage_detection_mitres(
        backend=backend,
        mitres=compiled_mitres
    )
    n_shots = 50
    result_list = leakage_gadget_mitres.run(
        [CircuitShots(Circuit=circuit, Shots=n_shots)]
    )
    counts = result_list[0].get_counts()
    assert all(shot in list(counts.keys()) for shot in [(0, 0), (1, 0)])
    assert sum(val for val in counts.values()) <= n_shots


def test_compare_with_prune() -> None:
    # A test to check for updates to prune that we should know about.

    circuit_0 = Circuit(2).measure_all()
    circuit_1 = Circuit(2).Rz(0.3, 0).measure_all()
    circuit_shot_0 = CircuitShots(Circuit=circuit_0, Shots=10)
    circuit_shot_1 = CircuitShots(Circuit=circuit_1, Shots=20)

    backend = MockQuantinuumBackend()

    generation_task = gen_add_leakage_gadget_circuit_task(backend=backend)
    postselection_task = gen_postselect_task()

    detection_circuit_shots_list, postselect_mgr_list = generation_task(
        ([circuit_shot_0, circuit_shot_1], )
    )
    result_list = [
        backend.run_circuit(
            circuit=backend.get_compiled_circuit(
                circuit=detection_circuit_shots.Circuit,
                optimisation_level=0
            ),
            n_shots=detection_circuit_shots.Shots,
        ) for detection_circuit_shots in detection_circuit_shots_list
    ]

    qermit_result_list = postselection_task(
        (result_list, postselect_mgr_list, )
    )
    pytket_result_list = [
        prune_shots_detected_as_leaky(result) for result in result_list
    ]
    assert qermit_result_list[0] == pytket_result_list

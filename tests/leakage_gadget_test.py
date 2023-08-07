from qermit.mock_backend import MockQuantinuumBackend
from pytket import Circuit
from qermit.taskgraph import gen_compiled_MitRes
from qermit import CircuitShots
from qermit.leakage_detection import get_leakage_detection_mitres


def test_leakage_gadget():

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

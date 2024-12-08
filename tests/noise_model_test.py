import json
import multiprocessing as mp
from collections import Counter
from copy import deepcopy
from itertools import product

import numpy as np
import pytest
from pytket import Circuit, OpType
from pytket.circuit import Qubit
from pytket.pauli import Pauli, QubitPauliString

from qermit.noise_model import (
    ErrorDistribution,
    LogicalErrorDistribution,
    NoiseModel,
    PauliErrorTranspile,
    QermitPauli,
    TranspilerBackend,
)
from qermit.noise_model.noise_model import Direction


def test_to_ptm() -> None:
    # A simple test with an only X noise model on one qubit
    error_distribution = ErrorDistribution(distribution={(Pauli.X,): 0.1})
    ptm, pauli_index = error_distribution.to_ptm()

    assert ptm[pauli_index[(Pauli.I,)]][pauli_index[(Pauli.I,)]] == 1
    assert ptm[pauli_index[(Pauli.X,)]][pauli_index[(Pauli.X,)]] == 1
    assert ptm[pauli_index[(Pauli.Y,)]][pauli_index[(Pauli.Y,)]] == 0.8
    assert ptm[pauli_index[(Pauli.Z,)]][pauli_index[(Pauli.Z,)]] == 0.8

    # A slightly more complicated example with some verified entries.
    error_distribution = ErrorDistribution(
        distribution={
            (Pauli.X, Pauli.Z): 0.08,
            (Pauli.Y, Pauli.Z): 0.02,
        }
    )

    ptm, pauli_index = error_distribution.to_ptm()

    assert abs(
        ptm[pauli_index[(Pauli.I, Pauli.I)]][pauli_index[(Pauli.I, Pauli.I)]] - 1
    ) < 10 ** (-6)
    assert abs(
        ptm[pauli_index[(Pauli.Z, Pauli.Z)]][pauli_index[(Pauli.Z, Pauli.Z)]] - 0.8
    ) < 10 ** (-6)
    assert abs(
        ptm[pauli_index[(Pauli.X, Pauli.Z)]][pauli_index[(Pauli.X, Pauli.Z)]] - 0.96
    ) < 10 ** (-6)
    assert abs(
        ptm[pauli_index[(Pauli.X, Pauli.X)]][pauli_index[(Pauli.X, Pauli.X)]] - 0.84
    ) < 10 ** (-6)


def test_from_ptm() -> None:
    # Test that the error distribution to and from ptm is the same as the initial
    distribution: dict[tuple[Pauli, ...], float] = {
        (Pauli.X, Pauli.X): 0.1,
        (Pauli.Y, Pauli.Z): 0.2,
        (Pauli.Z, Pauli.X): 0.3,
    }

    error_distribution = ErrorDistribution(
        distribution=distribution,
    )

    ptm, pauli_index = error_distribution.to_ptm()
    recovered_error_distribution = ErrorDistribution.from_ptm(
        ptm=ptm, pauli_index=pauli_index
    )

    for pauli, error_rate in distribution.items():
        assert abs(
            recovered_error_distribution.distribution[pauli] - error_rate
        ) < 10 ** (-6)

    # Test that from ptm is robust to moving Pauli indices
    (
        ptm[pauli_index[(Pauli.Y, Pauli.Z)]][pauli_index[(Pauli.Y, Pauli.Z)]],
        ptm[pauli_index[(Pauli.Z, Pauli.X)]][pauli_index[(Pauli.Z, Pauli.X)]],
    ) = (
        ptm[pauli_index[(Pauli.Z, Pauli.X)]][pauli_index[(Pauli.Z, Pauli.X)]],
        ptm[pauli_index[(Pauli.Y, Pauli.Z)]][pauli_index[(Pauli.Y, Pauli.Z)]],
    )
    pauli_index[(Pauli.Y, Pauli.Z)], pauli_index[(Pauli.Z, Pauli.X)] = (
        pauli_index[(Pauli.Z, Pauli.X)],
        pauli_index[(Pauli.Y, Pauli.Z)],
    )
    recovered_error_distribution = ErrorDistribution.from_ptm(
        ptm=ptm, pauli_index=pauli_index
    )

    for pauli, error_rate in distribution.items():
        assert abs(
            recovered_error_distribution.distribution[pauli] - error_rate
        ) < 10 ** (-6)


def test_qermit_pauli_from_iterable() -> None:
    qubit_pauli_string = QubitPauliString(
        qubits=[Qubit(i) for i in range(5)],
        paulis=[Pauli.X, Pauli.Y, Pauli.Z, Pauli.Y, Pauli.Z],
    )
    pauli = QermitPauli.from_pauli_iterable(
        pauli_iterable=qubit_pauli_string.map.values(),
        qubit_list=list(qubit_pauli_string.map.keys()),
    )
    pauli.qubit_pauli_string == (qubit_pauli_string, 1 + 0j)


def test_qermit_pauli_commute_coeff() -> None:
    # This tests a few commutation coefficients
    # which have been verified by hand.
    verified_list = [
        # Single qubit Paulis
        ((([0], [1]), ([1], [0])), -1),
        ((([1], [1]), ([1], [1])), 1),
        ((([1], [0]), ([0], [1])), -1),
        ((([0], [0]), ([0], [1])), 1),
        ((([1], [0]), ([0], [0])), 1),
        ((([1], [1]), ([0], [1])), -1),
        # Two qubit Paulis
        ((([0, 1], [1, 0]), ([1, 0], [0, 1])), 1),
        ((([0, 1], [1, 0]), ([0, 0], [1, 1])), -1),
        ((([0, 0], [0, 0]), ([0, 0], [1, 1])), 1),
    ]

    for verified in verified_list:
        n_qubits = len(verified[0][0][0])

        pauli_one = QermitPauli(
            Z_list=verified[0][0][0],
            X_list=verified[0][0][1],
            qubit_list=[Qubit(i) for i in range(n_qubits)],
        )
        pauli_two = QermitPauli(
            Z_list=verified[0][1][0],
            X_list=verified[0][1][1],
            qubit_list=[Qubit(i) for i in range(n_qubits)],
        )
        assert (
            QermitPauli.commute_coeff(pauli_one=pauli_one, pauli_two=pauli_two)
            == verified[1]
        )


def test_noise_model_logical_error_propagation() -> None:
    pytket_ciruit = Circuit(2).H(0).CX(0, 1).measure_all()

    error_distribution_dict: dict[tuple[Pauli, ...], float] = {
        (Pauli.X, Pauli.I): 0.1,
    }
    error_distribution = ErrorDistribution(
        distribution=error_distribution_dict,
        rng=np.random.default_rng(seed=0),
    )

    noise_model = NoiseModel({OpType.CX: error_distribution})
    logical_distribution = noise_model.get_effective_pre_error_distribution(
        pytket_ciruit, n_rand=10000
    )

    ideal_error_one = QubitPauliString(map={Qubit(0): Pauli.Z, Qubit(1): Pauli.X})
    assert list(logical_distribution.distribution.keys()) == [ideal_error_one]
    assert abs(logical_distribution.distribution[ideal_error_one] - 0.1) <= 0.01

    error_counter = noise_model.counter_propagate(
        pytket_ciruit, n_counts=10000, direction=Direction.forward
    )
    ideal_error_two = QermitPauli.from_qubit_pauli_string(
        QubitPauliString(map={Qubit(0): Pauli.X, Qubit(1): Pauli.I})
    )
    assert list(error_counter.keys()) == [ideal_error_two]
    assert abs(error_counter[ideal_error_two] - 1000) <= 1


def test_error_distribution_utilities(tmp_path_factory) -> None:
    # Test that the probabilities must be less than 1.
    error_distribution_dict: dict[tuple[Pauli, ...], float] = {(Pauli.X, Pauli.I): 1.1}
    with pytest.raises(Exception):
        error_distribution = ErrorDistribution(
            distribution=error_distribution_dict,
            rng=np.random.default_rng(seed=0),
        )

    # Test that averaging works as expected.
    error_distribution_dict = {
        (Pauli.X, Pauli.I): 0.1,
        (Pauli.I, Pauli.Z): 0.1,
    }
    error_distribution_one = ErrorDistribution(
        distribution=error_distribution_dict,
        rng=np.random.default_rng(seed=0),
    )

    error_distribution_dict = {
        (Pauli.X, Pauli.I): 0.1,
        (Pauli.Z, Pauli.I): 0.1,
    }
    error_distribution_two = ErrorDistribution(
        distribution=error_distribution_dict,
        rng=np.random.default_rng(seed=0),
    )

    error_distribution = ErrorDistribution.mixture(
        [error_distribution_one, error_distribution_two]
    )

    # Test that equality spots differences
    assert error_distribution != ErrorDistribution(
        distribution={
            (Pauli.X, Pauli.I): 0.1,
            (Pauli.Z, Pauli.I): 0.05,
        }
    )

    assert error_distribution != ErrorDistribution(
        distribution={
            (Pauli.X, Pauli.I): 0.1,
            (Pauli.Z, Pauli.I): 0.05,
            (Pauli.I, Pauli.Y): 0.05,
        }
    )

    assert error_distribution != ErrorDistribution(
        distribution={
            (Pauli.X, Pauli.I): 0.05,
            (Pauli.Z, Pauli.I): 0.05,
            (Pauli.I, Pauli.Z): 0.05,
        }
    )

    assert error_distribution == ErrorDistribution(
        distribution={
            (Pauli.X, Pauli.I): 0.1,
            (Pauli.Z, Pauli.I): 0.05,
            (Pauli.I, Pauli.Z): 0.05,
        }
    )

    error_distribution.order(reverse=False)
    assert list(error_distribution.distribution)[-1] == (Pauli.X, Pauli.I)

    # Test that distribution can be saved a loaded
    dist_path = tmp_path_factory.mktemp("distribution") / "dist.json"
    with dist_path.open(mode="w") as fp:
        json.dump(error_distribution.to_dict(), fp)

    with dist_path.open(mode="r") as fp:
        error_distribution_loaded = ErrorDistribution.from_dict(json.load(fp))

    assert error_distribution_loaded == error_distribution


def test_error_distribution_post_select() -> None:
    qps_remove = QubitPauliString(
        qubits=[
            Qubit(name="ancilla", index=0),
            Qubit(name="ancilla", index=1),
            Qubit(name="compute", index=0),
        ],
        paulis=[Pauli.X, Pauli.Z, Pauli.Y],
    )
    qps_keep = QubitPauliString(
        qubits=[
            Qubit(name="ancilla", index=0),
            Qubit(name="ancilla", index=1),
            Qubit(name="compute", index=0),
        ],
        paulis=[Pauli.I, Pauli.Z, Pauli.Y],
    )
    pauli_error_counter = Counter(
        {
            QermitPauli.from_qubit_pauli_string(qps_remove): 50,
            QermitPauli.from_qubit_pauli_string(qps_keep): 50,
        }
    )
    error_distribution = LogicalErrorDistribution(
        pauli_error_counter=pauli_error_counter
    )
    post_selected = error_distribution.post_select(
        qubit_list=[Qubit(name="ancilla", index=0), Qubit(name="ancilla", index=1)]
    )
    assert post_selected.distribution == {
        QubitPauliString(qubits=[Qubit(name="compute", index=0)], paulis=[Pauli.Y]): 1
    }


def test_to_dict(tmpdir_factory) -> None:
    error_distribution_dict_zzmax: dict[tuple[Pauli, ...], float] = {}
    error_distribution_dict_zzmax[(Pauli.X, Pauli.I)] = 0.0002
    error_distribution_dict_zzmax[(Pauli.I, Pauli.X)] = 0.0002
    error_distribution_dict_zzmax[(Pauli.I, Pauli.I)] = 0.9996

    error_distribution_dict_cz: dict[tuple[Pauli, ...], float] = {}
    error_distribution_dict_cz[(Pauli.Z, Pauli.Z)] = 0.002
    error_distribution_dict_cz[(Pauli.I, Pauli.Z)] = 0.002
    error_distribution_dict_cz[(Pauli.I, Pauli.I)] = 0.996

    error_distribution_cz = ErrorDistribution(
        error_distribution_dict_cz, rng=np.random.default_rng(seed=0)
    )
    error_distribution_zzmax = ErrorDistribution(
        error_distribution_dict_zzmax, rng=np.random.default_rng(seed=0)
    )

    noise_model = NoiseModel(
        {
            OpType.ZZMax: error_distribution_zzmax,
            OpType.CZ: error_distribution_cz,
        }
    )

    noise_model_dict = noise_model.to_dict()

    temp_dir = tmpdir_factory.mktemp("artifact")
    file_name = temp_dir.join("/noise_model.json")

    with open(file_name, "w") as fp:
        json.dump(noise_model_dict, fp)

    with open(file_name, "r") as fp:
        retrieved_noise_model_dict = json.load(fp)

    new_noise_model = NoiseModel.from_dict(retrieved_noise_model_dict)

    assert new_noise_model == noise_model


@pytest.mark.high_compute
def test_transpiler_backend() -> None:
    mp.set_start_method("spawn", force=True)

    circuit = Circuit(3)
    for _ in range(32):
        circuit.ZZMax(0, 1).ZZMax(1, 2)
    circuit.measure_all()

    error_distribution_dict: dict[tuple[Pauli, ...], float] = {}
    error_distribution_dict[(Pauli.X, Pauli.I)] = 0.002
    error_distribution_dict[(Pauli.I, Pauli.X)] = 0.002

    error_distribution_dict[(Pauli.I, Pauli.I)] = 0.996

    error_distribution = ErrorDistribution(
        distribution=error_distribution_dict,
        rng=np.random.default_rng(seed=0),
    )
    noise_model = NoiseModel({OpType.ZZMax: error_distribution})

    n_shots = 123
    transpiler = PauliErrorTranspile(noise_model=noise_model)
    backend = TranspilerBackend(transpiler=transpiler)

    handle = backend.process_circuit(circuit=circuit, n_shots=n_shots)
    result = backend.get_result(handle)

    assert sum(result.get_counts().values()) == n_shots


def test_pauli_error_transpile() -> None:
    error_distribution_dict = {
        error: 0.0 for error in product([Pauli.I, Pauli.X, Pauli.Y, Pauli.Z], repeat=2)
    }

    error_rate = 0.5
    error_distribution_dict[(Pauli.X, Pauli.I)] = error_rate
    error_distribution_dict[(Pauli.I, Pauli.X)] = error_rate
    error_distribution_dict[(Pauli.I, Pauli.I)] = 1 - 2 * error_rate

    error_distribution = ErrorDistribution(
        error_distribution_dict, rng=np.random.default_rng(seed=2)
    )

    noise_model = NoiseModel({OpType.ZZMax: error_distribution})

    circ = Circuit(2).ZZMax(0, 1).measure_all()
    transpiled_circ = Circuit(2).ZZMax(0, 1).X(1, opgroup="noisy").measure_all()

    transpiler = PauliErrorTranspile(noise_model=noise_model)
    transpiler.apply(circ)
    assert transpiled_circ == circ


def test_noise_model() -> None:
    mp.set_start_method("spawn", force=True)

    error_distribution_dict: dict[tuple[Pauli, ...], float] = {}
    error_rate = 0.5
    error_distribution_dict[(Pauli.X, Pauli.I)] = error_rate
    error_distribution_dict[(Pauli.I, Pauli.X)] = error_rate

    error_distribution = ErrorDistribution(
        error_distribution_dict, rng=np.random.default_rng(seed=2)
    )

    noise_model = NoiseModel({OpType.CZ: error_distribution})
    transpiler = PauliErrorTranspile(
        noise_model=noise_model,
    )
    backend = TranspilerBackend(transpiler=transpiler, max_batch_size=2)

    circuit = Circuit(2).CZ(0, 1).measure_all()
    result = backend.run_circuit(circuit=circuit, n_shots=3)
    counts = result.get_counts()
    assert all(shot in [(1, 0), (0, 1)] for shot in list(counts.keys()))


def test_error_backpropagation() -> None:
    # This is a backwards propagation of two X errors through a circuit
    # containing 2 CZ gates.

    name = "my_reg"
    qubit_list = [Qubit(name=name, index=0), Qubit(name=name, index=1)]

    stabilise = QermitPauli(
        Z_list=[0] * len(qubit_list),
        X_list=[0] * len(qubit_list),
        qubit_list=qubit_list,
    )

    stabilise.pre_apply_pauli(pauli=Pauli.X, qubit=qubit_list[1])
    stabilise.apply_gate(op_type=OpType.CZ, qubits=qubit_list)
    stabilise.pre_apply_pauli(pauli=Pauli.X, qubit=qubit_list[0])
    stabilise.apply_gate(op_type=OpType.CZ, qubits=[qubit_list[1], qubit_list[0]])

    assert stabilise.Z_list == {qubit_list[0]: 0, qubit_list[1]: 1}
    assert stabilise.X_list == {qubit_list[0]: 1, qubit_list[1]: 1}
    assert stabilise.phase == 0

    stabilise = QermitPauli(
        Z_list=[0] * len(qubit_list),
        X_list=[0] * len(qubit_list),
        qubit_list=qubit_list,
    )

    stabilise.pre_apply_pauli(pauli=Pauli.X, qubit=qubit_list[1])
    stabilise.apply_gate(op_type=OpType.CZ, qubits=qubit_list)
    stabilise.apply_gate(op_type=OpType.X, qubits=[qubit_list[1]])
    stabilise.pre_apply_pauli(pauli=Pauli.Y, qubit=qubit_list[0])
    stabilise.apply_gate(op_type=OpType.CZ, qubits=[qubit_list[1], qubit_list[0]])

    assert stabilise.Z_list == {qubit_list[0]: 1, qubit_list[1]: 1}
    assert stabilise.X_list == {qubit_list[0]: 1, qubit_list[1]: 1}
    assert stabilise.phase == 1


def test_back_propagate_random_error() -> None:
    cliff_circ = Circuit(2).CZ(0, 1).X(1).CZ(1, 0)
    qubit_list = cliff_circ.qubits

    error_rate = 0.5
    error_distribution_dict: dict[tuple[Pauli, ...], float] = {}
    error_distribution_dict[(Pauli.X, Pauli.I)] = error_rate
    error_distribution_dict[(Pauli.I, Pauli.X)] = error_rate
    error_distribution_dict[(Pauli.I, Pauli.I)] = 1 - 2 * error_rate

    error_distribution = ErrorDistribution(
        error_distribution_dict, rng=np.random.default_rng(seed=0)
    )
    noise_model = NoiseModel({OpType.CZ: error_distribution})

    # error_sampler = ErrorSampler(noise_model=noise_model)

    pauli_error = noise_model.random_propagate(cliff_circ)

    assert pauli_error.Z_list == {qubit_list[0]: 0, qubit_list[1]: 1}
    assert pauli_error.X_list == {qubit_list[0]: 0, qubit_list[1]: 0}
    assert pauli_error.phase == 2


# TODO: check this test by hand. It also takes a long time to run, which
# is unfortunate. It's also probabilistic at present which is not ideal.


@pytest.mark.high_compute
def test_effective_error_distribution() -> None:
    cliff_circ = Circuit(2).CZ(0, 1).X(1).CZ(1, 0)
    qubits = cliff_circ.qubits

    error_rate = 0.5
    error_distribution_dict: dict[tuple[Pauli, ...], float] = {}
    error_distribution_dict[(Pauli.X, Pauli.I)] = error_rate
    error_distribution_dict[(Pauli.I, Pauli.X)] = error_rate

    error_distribution = ErrorDistribution(
        error_distribution_dict,
        rng=np.random.default_rng(seed=0),
    )
    noise_model = NoiseModel({OpType.CZ: error_distribution})

    logical_error_distribution = noise_model.get_effective_pre_error_distribution(
        cliff_circ, n_rand=10000
    )

    assert all(
        abs(count - 2500) < 100
        for count in logical_error_distribution.pauli_error_counter.values()
    )

    cliff_circ = Circuit()
    cliff_circ.add_q_register(name="my_reg", size=3)
    qubits = cliff_circ.qubits
    cliff_circ.CZ(qubits[0], qubits[1]).CZ(qubits[1], qubits[2])

    error_distribution_dict = {}
    error_distribution_dict[(Pauli.X, Pauli.I)] = 0.3
    error_distribution_dict[(Pauli.I, Pauli.X)] = 0.7

    error_distribution = ErrorDistribution(
        error_distribution_dict, rng=np.random.default_rng(seed=0)
    )
    noise_model = NoiseModel({OpType.CZ: error_distribution})

    effective_error_dist = noise_model.get_effective_pre_error_distribution(
        cliff_circ, n_rand=10000
    )
    assert (
        abs(
            effective_error_dist.pauli_error_counter[
                QermitPauli.from_qubit_pauli_string(
                    QubitPauliString(qubits=qubits, paulis=[Pauli.X, Pauli.I, Pauli.X])
                )
            ]
            - 2100
        )
        < 100
    )

    assert (
        abs(
            effective_error_dist.pauli_error_counter[
                QermitPauli.from_qubit_pauli_string(
                    QubitPauliString(qubits=qubits, paulis=[Pauli.Y, Pauli.Y, Pauli.Z])
                )
            ]
            - 900
        )
        < 100
    )

    assert (
        abs(
            effective_error_dist.pauli_error_counter[
                QermitPauli.from_qubit_pauli_string(
                    QubitPauliString(qubits=qubits, paulis=[Pauli.I, Pauli.I, Pauli.Z])
                )
            ]
            - 2100
        )
        < 100
    )

    pauli_error = QermitPauli.from_qubit_pauli_string(
        QubitPauliString(qubits=qubits, paulis=[Pauli.Z, Pauli.Y, Pauli.X])
    )
    pauli_error.phase = 2
    assert abs(effective_error_dist.pauli_error_counter[pauli_error] - 4900) < 100


def test_qermit_pauli_circuit() -> None:
    circ = Circuit(3)
    circ.CZ(0, 1).add_barrier([0, 2]).H(1).H(2).SWAP(1, 2).S(1).Y(0)

    L = QermitPauli(Z_list=[1, 1, 1], X_list=[0, 0, 0], qubit_list=circ.qubits)
    L_circ = L.circuit

    R = deepcopy(L)
    R.apply_circuit(circ)
    R_circ = R.circuit

    check_circ = Circuit()
    for qubit in circ.qubits:
        check_circ.add_qubit(id=qubit)

    check_circ.add_circuit(circuit=L_circ, qubits=L_circ.qubits)
    check_circ.add_circuit(circuit=circ, qubits=circ.qubits)
    check_circ.add_circuit(circuit=R_circ, qubits=R_circ.qubits)

    assert np.allclose(circ.get_unitary(), check_circ.get_unitary())


def test_initialisation() -> None:
    qubit_list = [Qubit(0), Qubit(1), Qubit(2)]
    pauli = QermitPauli(
        Z_list=[0, 0, 1],
        X_list=[0, 0, 0],
        qubit_list=qubit_list,
    )
    assert pauli.X_list == {qubit_list[0]: 0, qubit_list[1]: 0, qubit_list[2]: 0}
    assert pauli.Z_list == {qubit_list[0]: 0, qubit_list[1]: 0, qubit_list[2]: 1}
    assert pauli.phase == 0


def test_identity_clifford() -> None:
    circ = Circuit(2).CZ(1, 0).X(0).S(1).X(0).X(1).H(1).X(1)
    circ.X(1).H(1).X(1).X(0).Sdg(1).X(0).CZ(1, 0)

    qubit_list = circ.qubits
    pauli = QermitPauli(
        Z_list=circ.n_qubits * [1], X_list=circ.n_qubits * [0], qubit_list=qubit_list
    )
    pauli.apply_circuit(circ)

    qubit_pauli_string, phase = pauli.qubit_pauli_string
    assert qubit_pauli_string == QubitPauliString(
        qubits=qubit_list, paulis=[Pauli.Z, Pauli.Z]
    )
    assert phase == 1


def test_H() -> None:
    qubit_list = [Qubit(0)]
    pauli = QermitPauli(Z_list=[1], X_list=[0], qubit_list=qubit_list)

    pauli.H(qubit=qubit_list[0])
    qubit_pauli_string, phase = pauli.qubit_pauli_string
    assert qubit_pauli_string == QubitPauliString(qubits=qubit_list, paulis=[Pauli.X])
    assert phase == 1

    pauli.H(qubit=qubit_list[0])
    qubit_pauli_string, phase = pauli.qubit_pauli_string
    assert qubit_pauli_string == QubitPauliString(qubits=qubit_list, paulis=[Pauli.Z])
    assert phase == 1


def test_h_series_gates() -> None:
    circ = Circuit(2).ZZPhase(3.5, 0, 1).PhasedX(1.5, 0.5, 0)
    stab = QermitPauli(Z_list=[1, 1], X_list=[1, 1], qubit_list=circ.qubits)
    stab.apply_circuit(circ)


def test_apply_circuit() -> None:
    circ = Circuit(2).H(0).S(0).CX(0, 1)
    qubit_list = circ.qubits
    pauli = QermitPauli(
        Z_list=circ.n_qubits * [1], X_list=circ.n_qubits * [0], qubit_list=qubit_list
    )

    pauli.apply_circuit(circ)

    qubit_pauli_string, phase = pauli.qubit_pauli_string
    assert qubit_pauli_string == QubitPauliString(
        qubits=qubit_list, paulis=[Pauli.X, Pauli.Y]
    )
    assert phase == 1


def test_apply_gate() -> None:
    qubit_list = [Qubit(0), Qubit(1), Qubit(2)]
    pauli = QermitPauli(
        Z_list=[1, 1, 1],
        X_list=[0, 0, 0],
        qubit_list=qubit_list,
    )

    pauli.apply_gate(op_type=OpType.H, qubits=[qubit_list[0]])

    qermit_qubit_pauli_string, phase = pauli.qubit_pauli_string
    assert qermit_qubit_pauli_string == QubitPauliString(
        qubits=qubit_list, paulis=[Pauli.X, Pauli.Z, Pauli.Z]
    )
    assert phase == 1

    pauli.apply_gate(
        op_type=OpType.CX,
        qubits=[qubit_list[1], qubit_list[2]],
    )

    qubit_pauli_string = QubitPauliString(
        qubits=qubit_list, paulis=[Pauli.X, Pauli.I, Pauli.Z]
    )
    qermit_qubit_pauli_string, phase = pauli.qubit_pauli_string
    assert qermit_qubit_pauli_string == qubit_pauli_string
    assert phase == 1

    pauli.apply_gate(op_type=OpType.S, qubits=[qubit_list[1]])
    qermit_qubit_pauli_string, phase = pauli.qubit_pauli_string

    assert qermit_qubit_pauli_string == qubit_pauli_string
    assert phase == 1


def test_qubit_pauli_string() -> None:
    qubit_list = [Qubit(0), Qubit(1), Qubit(2)]
    pauli = QermitPauli(
        Z_list=[1, 1, 1],
        X_list=[0, 0, 0],
        qubit_list=qubit_list,
    )

    qubit_pauli_string = QubitPauliString(
        qubits=qubit_list, paulis=[Pauli.Z, Pauli.Z, Pauli.Z]
    )
    qermit_qubit_pauli_string, phase = pauli.qubit_pauli_string
    assert qermit_qubit_pauli_string == qubit_pauli_string
    assert phase == 1

    pauli.H(qubit_list[0])
    pauli.S(qubit_list[0])

    qubit_pauli_string = QubitPauliString(
        qubits=qubit_list, paulis=[Pauli.Y, Pauli.Z, Pauli.Z]
    )
    qermit_qubit_pauli_string, phase = pauli.qubit_pauli_string
    assert qermit_qubit_pauli_string == qubit_pauli_string
    assert phase == 1

    pauli.CX(qubit_list[0], qubit_list[1])

    qubit_pauli_string = QubitPauliString(
        qubits=qubit_list, paulis=[Pauli.X, Pauli.Y, Pauli.Z]
    )
    qermit_qubit_pauli_string, phase = pauli.qubit_pauli_string
    assert qermit_qubit_pauli_string == qubit_pauli_string
    assert phase == 1

    pauli.S(qubit_list[1])

    qubit_pauli_string = QubitPauliString(
        qubits=qubit_list, paulis=[Pauli.X, Pauli.X, Pauli.Z]
    )
    qermit_qubit_pauli_string, phase = pauli.qubit_pauli_string
    assert qermit_qubit_pauli_string == qubit_pauli_string
    assert phase == -1

    pauli.S(qubit_list[0])
    pauli.CX(qubit_list[0], qubit_list[2])

    qubit_pauli_string = QubitPauliString(
        qubits=qubit_list, paulis=[Pauli.X, Pauli.X, Pauli.Y]
    )
    qermit_qubit_pauli_string, phase = pauli.qubit_pauli_string
    assert qermit_qubit_pauli_string == qubit_pauli_string
    assert phase == -1


def test_clifford_incremental() -> None:
    qubit_list = [Qubit(0), Qubit(1), Qubit(2)]
    pauli = QermitPauli(
        Z_list=[0, 0, 1],
        X_list=[0, 0, 0],
        qubit_list=qubit_list,
    )

    pauli.H(qubit_list[0])
    assert pauli.X_list == {qubit_list[0]: 0, qubit_list[1]: 0, qubit_list[2]: 0}
    assert pauli.Z_list == {qubit_list[0]: 0, qubit_list[1]: 0, qubit_list[2]: 1}
    assert pauli.phase == 0

    pauli.CX(qubit_list[1], qubit_list[2])
    assert pauli.X_list == {qubit_list[0]: 0, qubit_list[1]: 0, qubit_list[2]: 0}
    assert pauli.Z_list == {qubit_list[0]: 0, qubit_list[1]: 1, qubit_list[2]: 1}
    assert pauli.phase == 0

    pauli.H(qubit_list[1])
    assert pauli.X_list == {qubit_list[0]: 0, qubit_list[1]: 1, qubit_list[2]: 0}
    assert pauli.Z_list == {qubit_list[0]: 0, qubit_list[1]: 0, qubit_list[2]: 1}
    assert pauli.phase == 0

    pauli.S(qubit_list[1])
    assert pauli.X_list == {qubit_list[0]: 0, qubit_list[1]: 1, qubit_list[2]: 0}
    assert pauli.Z_list == {qubit_list[0]: 0, qubit_list[1]: 1, qubit_list[2]: 1}
    assert pauli.phase == 1

    pauli.CX(qubit_list[1], qubit_list[2])
    assert pauli.X_list == {qubit_list[0]: 0, qubit_list[1]: 1, qubit_list[2]: 1}
    assert pauli.Z_list == {qubit_list[0]: 0, qubit_list[1]: 0, qubit_list[2]: 1}
    assert pauli.phase == 1

    pauli.S(qubit_list[2])
    assert pauli.X_list == {qubit_list[0]: 0, qubit_list[1]: 1, qubit_list[2]: 1}
    assert pauli.Z_list == {qubit_list[0]: 0, qubit_list[1]: 0, qubit_list[2]: 0}
    assert pauli.phase == 2

    pauli.CX(qubit_list[1], qubit_list[0])
    assert pauli.X_list == {qubit_list[0]: 1, qubit_list[1]: 1, qubit_list[2]: 1}
    assert pauli.Z_list == {qubit_list[0]: 0, qubit_list[1]: 0, qubit_list[2]: 0}
    assert pauli.phase == 2

    pauli.S(qubit_list[0])
    assert pauli.X_list == {qubit_list[0]: 1, qubit_list[1]: 1, qubit_list[2]: 1}
    assert pauli.Z_list == {qubit_list[0]: 1, qubit_list[1]: 0, qubit_list[2]: 0}
    assert pauli.phase == 3

    pauli.H(qubit_list[0])
    assert pauli.X_list == {qubit_list[0]: 1, qubit_list[1]: 1, qubit_list[2]: 1}
    assert pauli.Z_list == {qubit_list[0]: 1, qubit_list[1]: 0, qubit_list[2]: 0}
    assert pauli.phase == 1


def test_to_from_qps() -> None:
    qubits = [Qubit(name=f"reg_name_{i}", index=0) for i in range(5)]
    paulis = [Pauli.I, Pauli.Y, Pauli.X, Pauli.Y, Pauli.Z]
    qubit_pauli_string = QubitPauliString(
        qubits=qubits,
        paulis=paulis,
    )
    stab = QermitPauli.from_qubit_pauli_string(qubit_pauli_string)
    stab_qps, stab_phase = stab.qubit_pauli_string
    assert stab_qps == qubit_pauli_string
    assert stab_phase == 1 + 0j


def test_is_measureable() -> None:
    stab = QermitPauli(
        Z_list=[1, 0, 0],
        X_list=[1, 0, 1],
        qubit_list=[
            Qubit(name="A", index=0),
            Qubit(name="B", index=0),
            Qubit(name="A", index=1),
        ],
    )
    assert stab.is_measureable(qubit_list=[Qubit(name="A", index=1)])
    assert stab.is_measureable(qubit_list=[Qubit(name="A", index=0)])
    assert not stab.is_measureable(qubit_list=[Qubit(name="B", index=0)])


def test_noise_model_scaling() -> None:
    # Here we test a couple of hand calculated examples.
    error_distribution_cz = ErrorDistribution(
        distribution={
            (Pauli.X, Pauli.I): 0.1,
            (Pauli.Z, Pauli.I): 0.01,
        }
    )
    error_distribution_cx = ErrorDistribution(
        distribution={
            (Pauli.X, Pauli.Y): 0.2,
            (Pauli.Z, Pauli.X): 0.3,
        }
    )
    noise_model = NoiseModel(
        noise_model={OpType.CZ: error_distribution_cz, OpType.CX: error_distribution_cx}
    )

    # In the zero folding case we should end up with an empty noise model.
    zero_scaled_noise_model = noise_model.scale(scaling_factor=0)
    assert zero_scaled_noise_model.noise_model[OpType.CX].distribution == {}
    assert zero_scaled_noise_model.noise_model[OpType.CZ].distribution == {}

    # In the two folded case, we first check the PTM on the noise
    # on the CZ gate. Note that the PTM values for the original
    # error channel are being squared in this case.
    two_scaled_noise_model = noise_model.scale(scaling_factor=2)
    ptm, pauli_index = two_scaled_noise_model.noise_model[OpType.CZ].to_ptm()
    for pauli_one, ptm_entry in zip(
        [Pauli.I, Pauli.X, Pauli.Y, Pauli.Z], [1, 0.98**2, 0.78**2, 0.8**2]
    ):
        assert all(
            ptm[pauli_index[(pauli_one, pauli_two)]][
                pauli_index[(pauli_one, pauli_two)]
            ]
            == ptm_entry
            for pauli_two in [Pauli.I, Pauli.X, Pauli.Y, Pauli.Z]
        )

    # Here we work through pre computed values for the
    # error rates.
    assert list(two_scaled_noise_model.noise_model[OpType.CZ].distribution.keys()) == [
        (Pauli.X, Pauli.I),
        (Pauli.Y, Pauli.I),
        (Pauli.Z, Pauli.I),
    ]
    assert abs(
        two_scaled_noise_model.noise_model[OpType.CZ].distribution[(Pauli.X, Pauli.I)]
        - 0.178
    ) < 10 ** (-6)
    assert abs(
        two_scaled_noise_model.noise_model[OpType.CZ].distribution[(Pauli.Y, Pauli.I)]
        - 0.002
    ) < 10 ** (-6)
    assert abs(
        two_scaled_noise_model.noise_model[OpType.CZ].distribution[(Pauli.Z, Pauli.I)]
        - 0.0178
    ) < 10 ** (-6)

    assert list(two_scaled_noise_model.noise_model[OpType.CX].distribution.keys()) == [
        (Pauli.X, Pauli.Y),
        (Pauli.Y, Pauli.Z),
        (Pauli.Z, Pauli.X),
    ]
    assert abs(
        two_scaled_noise_model.noise_model[OpType.CX].distribution[(Pauli.X, Pauli.Y)]
        - 0.2
    ) < 10 ** (-6)
    assert abs(
        two_scaled_noise_model.noise_model[OpType.CX].distribution[(Pauli.Y, Pauli.Z)]
        - 0.12
    ) < 10 ** (-6)
    assert abs(
        two_scaled_noise_model.noise_model[OpType.CX].distribution[(Pauli.Z, Pauli.X)]
        - 0.3
    ) < 10 ** (-6)


if __name__ == "__main__":
    test_is_measureable()
    test_to_from_qps()
    test_clifford_incremental()
    test_qubit_pauli_string()
    test_apply_gate()
    test_apply_circuit()
    test_h_series_gates()
    test_H()
    test_identity_clifford()
    test_initialisation()
    test_qermit_pauli_circuit()
    test_effective_error_distribution()
    test_back_propagate_random_error()
    test_error_backpropagation()
    test_noise_model()
    test_pauli_error_transpile()
    test_transpiler_backend()
    test_error_distribution_post_select()
    test_noise_model_logical_error_propagation()

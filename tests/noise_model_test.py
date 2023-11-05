from itertools import product
from pytket.pauli import Pauli
from pytket import Circuit, OpType
from qermit.mock_backend import (
    PauliErrorTranspile,
    TranspilerBackend,
    NoiseModel,
    ErrorDistribution,
    ErrorSampler,
)
from collections import Counter
from pytket.circuit import Qubit
from qermit.mock_backend.stabiliser import Stabiliser
from pytket.pauli import QubitPauliString, Pauli
import pytest
import json
import numpy as np


def test_error_distribution_post_select():

    qps_remove = QubitPauliString(
        qubits = [Qubit(name='ancilla', index=0), Qubit(name='ancilla', index=1), Qubit(name='compute', index=0)],
        paulis = [Pauli.X, Pauli.Z, Pauli.Y]
    )
    qps_keep = QubitPauliString(
        qubits = [Qubit(name='ancilla', index=0), Qubit(name='ancilla', index=1), Qubit(name='compute', index=0)],
        paulis = [Pauli.I, Pauli.Z, Pauli.Y]
    )
    distribution = {qps_remove:0.5, qps_keep:0.5}
    error_distribution = ErrorDistribution(distribution=distribution)
    post_selected = error_distribution.post_select(
        qubit_list = [Qubit(name='ancilla', index=0), Qubit(name='ancilla', index=1)]
    )
    assert post_selected == ErrorDistribution(
        distribution = {
                QubitPauliString(
                qubits = [Qubit(name='compute', index=0)],
                paulis = [Pauli.Y]
            ):0.5
        }
    )


def test_to_dict(tmpdir_factory):

    error_distribution_dict_zzmax = {}
    error_distribution_dict_zzmax[(Pauli.X, Pauli.I)] = 0.0002
    error_distribution_dict_zzmax[(Pauli.I, Pauli.X)] = 0.0002
    error_distribution_dict_zzmax[(Pauli.I, Pauli.I)] = 0.9996

    error_distribution_dict_cz = {}
    error_distribution_dict_cz[(Pauli.Z, Pauli.Z)] = 0.002
    error_distribution_dict_cz[(Pauli.I, Pauli.Z)] = 0.002
    error_distribution_dict_cz[(Pauli.I, Pauli.I)] = 0.996

    error_distribution_cz = ErrorDistribution(
        error_distribution_dict_cz, rng=np.random.default_rng(seed=0))
    error_distribution_zzmax = ErrorDistribution(
        error_distribution_dict_zzmax, rng=np.random.default_rng(seed=0))

    noise_model = NoiseModel(
        {
            OpType.ZZMax: error_distribution_zzmax,
            OpType.CZ: error_distribution_cz,
        }
    )

    noise_model_dict = noise_model.to_dict()

    temp_dir = tmpdir_factory.mktemp("artifact")
    file_name = temp_dir.join("/noise_model.json")

    with open(file_name, 'w') as fp:
        json.dump(noise_model_dict, fp)

    with open(file_name, 'r') as fp:
        retrieved_noise_model_dict = json.load(fp)

    new_noise_model = NoiseModel.from_dict(retrieved_noise_model_dict)

    assert new_noise_model == noise_model


def test_transpiler_backend():

    circuit = Circuit(3)
    for _ in range(32):
        circuit.ZZMax(0, 1).ZZMax(1, 2)
    circuit.measure_all()

    error_distribution_dict = {}
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


def test_pauli_error_transpile():

    error_distribution_dict = {
        error: 0 for error in product(
            [Pauli.I, Pauli.X, Pauli.Y, Pauli.Z], repeat=2
        )
    }

    error_rate = 0.5
    error_distribution_dict[(Pauli.X, Pauli.I)] = error_rate
    error_distribution_dict[(Pauli.I, Pauli.X)] = error_rate
    error_distribution_dict[(Pauli.I, Pauli.I)] = 1 - 2*error_rate
    # error_distribution_dict[(Pauli.I, Pauli.I)] = 1 - 2*error_rate

    error_distribution = ErrorDistribution(error_distribution_dict, rng=np.random.default_rng(seed=2))

    noise_model = NoiseModel({OpType.ZZMax: error_distribution})

    circ = Circuit(2).ZZMax(0, 1).measure_all()
    transpiled_circ = Circuit(2).ZZMax(0, 1).X(
        1, opgroup='noisy').measure_all()

    transpiler = PauliErrorTranspile(
        noise_model=noise_model
    )
    transpiler.apply(circ)
    assert transpiled_circ == circ


def test_noise_model():

    error_distribution_dict = {}
    error_rate = 0.5
    error_distribution_dict[(Pauli.X, Pauli.I)] = error_rate
    error_distribution_dict[(Pauli.I, Pauli.X)] = error_rate

    error_distribution = ErrorDistribution(error_distribution_dict, rng=np.random.default_rng(seed=2))

    noise_model = NoiseModel({OpType.CZ: error_distribution})
    transpiler = PauliErrorTranspile(
        noise_model=noise_model,
    )
    backend = TranspilerBackend(transpiler=transpiler)

    circuit = Circuit(2).CZ(0, 1).measure_all()
    counts = backend.get_counts(
        circuit=circuit,
        n_shots=1,
    )
    assert counts == Counter({(1, 0): 1})


# def test_error_backpropagation():

#     # This is a backwards propagation of two X errors through a circuit
#     # containing 2 CZ gates.

#     name = 'my_reg'
#     qubit_list = [Qubit(name=name, index=0), Qubit(name=name, index=1)]

#     stabilise = Stabiliser(
#         Z_list=[0] * len(qubit_list),
#         X_list=[0] * len(qubit_list),
#         qubit_list=qubit_list,
#     )

#     stabilise.pre_apply_pauli(pauli=Pauli.X, qubit=qubit_list[1])
#     stabilise.apply_gate(op_type=OpType.CZ, qubits=qubit_list)
#     stabilise.pre_apply_pauli(pauli=Pauli.X, qubit=qubit_list[0])
#     stabilise.apply_gate(op_type=OpType.CZ, qubits=[
#                          qubit_list[1], qubit_list[0]])

#     assert stabilise.Z_list == {qubit_list[0]: 0, qubit_list[1]: 1}
#     assert stabilise.X_list == {qubit_list[0]: 1, qubit_list[1]: 1}
#     assert stabilise.phase == 0

#     stabilise = Stabiliser(
#         Z_list=[0] * len(qubit_list),
#         X_list=[0] * len(qubit_list),
#         qubit_list=qubit_list,
#     )

#     stabilise.pre_apply_pauli(pauli=Pauli.X, qubit=qubit_list[1])
#     stabilise.apply_gate(op_type=OpType.CZ, qubits=qubit_list)
#     stabilise.apply_gate(op_type=OpType.X, qubits=[qubit_list[1]])
#     stabilise.pre_apply_pauli(pauli=Pauli.Y, qubit=qubit_list[0])
#     stabilise.apply_gate(op_type=OpType.CZ, qubits=[
#                          qubit_list[1], qubit_list[0]])

#     assert stabilise.Z_list == {qubit_list[0]: 1, qubit_list[1]: 1}
#     assert stabilise.X_list == {qubit_list[0]: 1, qubit_list[1]: 1}
#     assert stabilise.phase == 1


def test_back_propagate_random_error():

    cliff_circ = Circuit(2).CZ(0, 1).X(1).CZ(1, 0)
    qubit_list = cliff_circ.qubits

    error_rate = 0.5
    error_distribution_dict = {}
    error_distribution_dict[(Pauli.X, Pauli.I)] = error_rate
    error_distribution_dict[(Pauli.I, Pauli.X)] = error_rate
    error_distribution_dict[(Pauli.I, Pauli.I)] = 1 - 2*error_rate

    error_distribution = ErrorDistribution(
        error_distribution_dict,
        rng=np.random.default_rng(seed=0)
    )
    noise_model = NoiseModel({OpType.CZ: error_distribution})

    error_sampler = ErrorSampler(noise_model=noise_model)

    stabiliser = error_sampler.random_propagate(cliff_circ)

    assert stabiliser.Z_list == {qubit_list[0]: 0, qubit_list[1]: 1}
    assert stabiliser.X_list == {qubit_list[0]: 0, qubit_list[1]: 0}
    assert stabiliser.phase == 2

# TODO: check this test by hand. It also takes a long time to run, which
# is unfortunate. It's also probabilistic at present which is not ideal.


@pytest.mark.high_compute
def test_effective_error_distribution():

    cliff_circ = Circuit(2).CZ(0, 1).X(1).CZ(1, 0)
    qubits = cliff_circ.qubits

    error_rate = 0.5
    error_distribution_dict = {}
    error_distribution_dict[(Pauli.X, Pauli.I)] = error_rate
    error_distribution_dict[(Pauli.I, Pauli.X)] = error_rate
    error_distribution_dict[(Pauli.I, Pauli.I)] = 1 - 2*error_rate

    error_distribution = ErrorDistribution(
        error_distribution_dict,
        rng=np.random.default_rng(seed=0),
    )
    noise_model = NoiseModel({OpType.CZ: error_distribution})

    error_distribution = noise_model.get_effective_pre_error_distribution(
        cliff_circ, n_rand=100000)
    ideal_error_distribution = ErrorDistribution(
        distribution={
            QubitPauliString(qubits=qubits, paulis=[Pauli.I, Pauli.Z]): 0.25,
            QubitPauliString(qubits=qubits, paulis=[Pauli.X, Pauli.Y]): 0.25,
            QubitPauliString(qubits=qubits, paulis=[Pauli.Y, Pauli.X]): 0.25,
            QubitPauliString(qubits=qubits, paulis=[Pauli.Z, Pauli.I]): 0.25,
        }
    )
    assert error_distribution == ideal_error_distribution

    # I've checked this second half

    cliff_circ = Circuit()
    cliff_circ.add_q_register(name='my_reg', size=3)
    qubits = cliff_circ.qubits
    cliff_circ.CZ(qubits[0], qubits[1]).CZ(qubits[1], qubits[2])

    error_distribution_dict = {}
    error_distribution_dict[(Pauli.X, Pauli.I)] = 0.3
    error_distribution_dict[(Pauli.I, Pauli.X)] = 0.7

    error_distribution = ErrorDistribution(
        error_distribution_dict,
        rng=np.random.default_rng(seed=0)
    )
    noise_model = NoiseModel({OpType.CZ: error_distribution})

    effective_error_dist = noise_model.get_effective_pre_error_distribution(
        cliff_circ, n_rand=100000
    )
    ideal_error_distribution = ErrorDistribution(
        distribution={
            QubitPauliString(
                qubits=qubits, paulis=[Pauli.Y, Pauli.Y, Pauli.Z]
            ): 0.09,
            QubitPauliString(
                qubits=qubits, paulis=[Pauli.Z, Pauli.Y, Pauli.X]
            ): 0.49,
            QubitPauliString(
                qubits=qubits, paulis=[Pauli.X, Pauli.I, Pauli.X]
            ): 0.21,
            QubitPauliString(
                qubits=qubits, paulis=[Pauli.I, Pauli.I, Pauli.Z]
            ): 0.21,
        }
    )
    assert effective_error_dist == ideal_error_distribution

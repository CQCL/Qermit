# Copyright 2019-2021 Cambridge Quantum Computing
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from qermit.spam.full_transition_tomography import (  # type: ignore
    binary_to_int,
    int_to_binary,
    get_full_transition_tomography_circuits,
    calculate_correlation_matrices,
    correct_transition_noise,
    CorrectionMethod,
    reduce_matrix,
    reduce_matrices,
)

from pytket import Circuit, Bit
from pytket.extensions.qiskit import AerBackend  # type: ignore
import qiskit.providers.aer.noise as noise  # type: ignore
import numpy as np

# set up basic readout error noise model backend for tests
def get_noisy_backend(n_qubits, prob_ro):
    noise_model = noise.NoiseModel()
    probabilities = [[1 - prob_ro, prob_ro], [prob_ro, 1 - prob_ro]]
    error_ro = noise.ReadoutError(probabilities)
    for i in range(n_qubits):
        noise_model.add_readout_error(error_ro, [i])
    return AerBackend(noise_model)


def test_binary_int_methods():
    test_int = 2
    converted_binary = int_to_binary(test_int, 5)
    assert binary_to_int(converted_binary) == test_int


def test_get_transition_tomography_circuits():
    backend = get_noisy_backend(4, 0.1)
    pc = Circuit(4).CX(0, 1)
    nodes = backend.backend_info.architecture.nodes
    correlations = [[nodes[0], nodes[1]], [nodes[2], nodes[3]]]
    # get tomography circuits
    output = get_full_transition_tomography_circuits(pc, backend, correlations)

    assert len(output) == 2

    tomo_circs = output[0]
    tomo_dicts = output[1]

    assert len(tomo_circs) == len(tomo_dicts)

    # comparison dictionaries
    corr_dict_0 = {(nodes[0], nodes[1]): (0, 0), (nodes[2], nodes[3]): (0, 0)}
    corr_dict_1 = {(nodes[0], nodes[1]): (0, 1), (nodes[2], nodes[3]): (0, 1)}
    corr_dict_2 = {(nodes[0], nodes[1]): (1, 0), (nodes[2], nodes[3]): (1, 0)}
    corr_dict_3 = {(nodes[0], nodes[1]): (1, 1), (nodes[2], nodes[3]): (1, 1)}

    # qb bit maps
    qb_bit_map = {
        nodes[0]: Bit(0),
        nodes[1]: Bit(1),
        nodes[2]: Bit(2),
        nodes[3]: Bit(3),
    }
    # check that returned circuits have right number of commands, given extra commands
    # implies X gates for state prep in native gateset
    assert tomo_dicts[0][0] == corr_dict_0
    assert tomo_dicts[0][1] == qb_bit_map
    assert len(tomo_circs[0].get_commands()) == 7
    assert tomo_dicts[1][0] == corr_dict_1
    assert tomo_dicts[1][1] == qb_bit_map
    assert len(tomo_circs[1].get_commands()) == 9
    assert tomo_dicts[2][0] == corr_dict_2
    assert tomo_dicts[2][1] == qb_bit_map
    assert len(tomo_circs[2].get_commands()) == 9
    assert tomo_dicts[3][0] == corr_dict_3
    assert tomo_dicts[3][1] == qb_bit_map
    assert len(tomo_circs[3].get_commands()) == 11


def test_calculate_correlation_matrices():
    backend = get_noisy_backend(4, 0.0000001)
    pc = Circuit(4)
    nodes = backend.backend_info.architecture.nodes
    correlations_0 = [[nodes[0]], [nodes[1]], [nodes[2]], [nodes[3]]]
    correlations_1 = [[nodes[0], nodes[1]], [nodes[2], nodes[3]]]

    output_0 = get_full_transition_tomography_circuits(pc, backend, correlations_0)
    output_1 = get_full_transition_tomography_circuits(pc, backend, correlations_1)

    assert len(output_0[0]) == 2
    # sim results through near noiseless simulation
    handles_0 = backend.process_circuits(output_0[0], 5)
    results_0 = backend.get_results(handles_0)

    handles_1 = backend.process_circuits(output_1[0], 5)
    results_1 = backend.get_results(handles_1)

    pnc_0 = calculate_correlation_matrices(results_0, output_0[1], correlations_0)
    pnc_1 = calculate_correlation_matrices(results_1, output_1[1], correlations_1)

    # first element of PureNoiseCharacterisation is correlations characterisation is produced for
    assert pnc_0.CorrelatedNodes == correlations_0
    assert pnc_1.CorrelatedNodes == correlations_1

    # second element of PureNoiseCharacterisation is a dictionary from node to matrix indexing for any measured qubit
    # to allow sensible results retrieval
    assert pnc_0.NodeToIntDict == {
        nodes[0]: (0, 0),
        nodes[1]: (1, 0),
        nodes[2]: (2, 0),
        nodes[3]: (3, 0),
    }
    assert pnc_1.NodeToIntDict == {
        nodes[0]: (0, 0),
        nodes[1]: (0, 1),
        nodes[2]: (1, 0),
        nodes[3]: (1, 1),
    }

    # third element of PureNoiseCharacterisation are the characterisation matrices
    assert len(pnc_0.CharacterisationMatrices) == 4
    assert len(pnc_1.CharacterisationMatrices) == 2

    identity_2 = np.identity(2).all()
    identity_4 = np.identity(4).all()

    assert pnc_0.CharacterisationMatrices[0].all() == identity_2
    assert pnc_0.CharacterisationMatrices[1].all() == identity_2
    assert pnc_0.CharacterisationMatrices[2].all() == identity_2
    assert pnc_0.CharacterisationMatrices[3].all() == identity_2

    assert pnc_1.CharacterisationMatrices[0].all() == identity_4
    assert pnc_1.CharacterisationMatrices[1].all() == identity_4


def test_reduce_matrices():
    test_mat = np.zeros((4,) * 2, dtype=float)
    counter = 0
    for i in range(4):
        for j in range(4):
            test_mat[i, j] = i + j + counter
            counter += 1
    reduced_matrix_0 = reduce_matrix([0], test_mat)
    reduced_matrix_1 = reduce_matrix([1], test_mat)

    comp_0 = np.array(
        [[test_mat[0, 0], test_mat[0, 2]], [test_mat[2, 0], test_mat[2, 2]]]
    )
    comp_1 = np.array(
        [[test_mat[0, 0], test_mat[0, 1]], [test_mat[1, 0], test_mat[1, 1]]]
    )

    assert reduced_matrix_0.all() == comp_0.all()
    assert reduced_matrix_1.all() == comp_1.all()

    reduced_matrices = reduce_matrices([(0, 1), (1, 0)], [test_mat, test_mat])
    assert reduced_matrices[0].all() == reduced_matrix_1.all()
    assert reduced_matrices[1].all() == reduced_matrix_0.all()


def test_correct_transition_noise():
    backend = get_noisy_backend(4, 0.1)
    # get test circuit
    test_circuit = Circuit(4).X(0).X(1).X(2).X(3).measure_all()
    test_circuit = backend.get_compiled_circuit(test_circuit)
    qubit_readout = test_circuit.qubit_to_bit_map
    handles = backend.process_circuits([test_circuit], 50)
    results = backend.get_results(handles)

    # get test characterisation
    pc = Circuit(4)
    nodes = backend.backend_info.architecture.nodes
    correlations = [[nodes[0], nodes[1]], [nodes[2], nodes[3]]]
    tomo_circs = get_full_transition_tomography_circuits(pc, backend, correlations)
    tomo_handles = backend.process_circuits(tomo_circs[0], 5)
    tomo_results = backend.get_results(tomo_handles)
    pnc = calculate_correlation_matrices(tomo_results, tomo_circs[1], correlations)

    assert correct_transition_noise(
        results[0], (qubit_readout, {}), pnc, CorrectionMethod.Invert
    )
    assert correct_transition_noise(
        results[0], (qubit_readout, {}), pnc, CorrectionMethod.Bayesian
    )


if __name__ == "__main__":
    test_binary_int_methods()
    test_get_transition_tomography_circuits()
    test_calculate_correlation_matrices()
    test_correct_transition_noise()
    test_reduce_matrices()

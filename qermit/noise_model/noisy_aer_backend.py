# Copyright 2019-2023 Quantinuum
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

import qiskit_aer.noise as noise  # type: ignore
from pytket import OpType
from pytket.extensions.qiskit import AerBackend


class NoisyAerBackend(AerBackend):
    """AerBacked with simple depolarising and SPAM noise model. Depolarising
    noise is added to the gateset {OpType.CX, OpType.H, OpType.Rz,
    OpType.Rz, OpType.Measure} and circuits should be rebased into
    that gateset before running.
    """

    noisy_gate_set = {OpType.CX, OpType.H, OpType.Rz, OpType.Rz, OpType.Measure}

    def __init__(self, n_qubits: int, prob_1: float, prob_2: float, prob_ro: float):
        """AerBacked with simple depolarising and SPAM noise model.

        :param n_qubits: The number of qubits available on the backend.
        :param prob_1: The depolarising noise error rates on single qubit gates.
        :param prob_2: The depolarising noise error rates on two qubit gates.
        :param prob_ro: Error rates of symmetric uncorrelated SPAM errors.
        """

        super().__init__(
            noise_model=self.depolarizing_noise_model(n_qubits, prob_1, prob_2, prob_ro)
        )

    def depolarizing_noise_model(
        self,
        n_qubits: int,
        prob_1: float,
        prob_2: float,
        prob_ro: float,
    ) -> noise.NoiseModel:
        """Generates noise model, may be passed to `noise_model` parameter of
        AerBacked.

        :param n_qubits: Number of qubits noise model applies to.
        :param prob_1: The depolarising noise error rates on single qubit gates.
        :param prob_2: The depolarising noise error rates on two qubit gates.
        :param prob_ro: Error rates of symmetric uncorrelated SPAM errors.
        :return: Noise model
        """

        noise_model = noise.NoiseModel()

        error_2 = noise.depolarizing_error(prob_2, 2)
        for edge in [[i, j] for i in range(n_qubits) for j in range(i)]:
            noise_model.add_quantum_error(error_2, ["cx"], [edge[0], edge[1]])
            noise_model.add_quantum_error(error_2, ["cx"], [edge[1], edge[0]])

        error_1 = noise.depolarizing_error(prob_1, 1)
        for node in range(n_qubits):
            noise_model.add_quantum_error(error_1, ["h", "rx", "rz"], [node])

        probabilities = [[1 - prob_ro, prob_ro], [prob_ro, 1 - prob_ro]]
        error_ro = noise.ReadoutError(probabilities)
        for i in range(n_qubits):
            noise_model.add_readout_error(error_ro, [i])

        return noise_model

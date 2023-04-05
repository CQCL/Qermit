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

from pytket import OpType
from pytket.backends.backendinfo import BackendInfo
from pytket.architecture import FullyConnected  # type: ignore
from pytket.passes.auto_rebase import auto_rebase_pass
from pytket.predicates import GateSetPredicate  # type: ignore
from pytket.extensions.quantinuum import QuantinuumBackend  # type: ignore
from pytket.extensions.quantinuum.backends.quantinuum import _GATE_SET  # type: ignore
from pytket.predicates import CompilationUnit  # type: ignore
from pytket.extensions.qiskit import AerBackend  # type: ignore
import qiskit.providers.aer.noise as noise  # type: ignore
from pytket import OpType
from pytket import Circuit
from pytket.backends.resulthandle import ResultHandle
from typing import List, Union
from pytket.backends.backendresult import BackendResult


class NoisyAerBackend(AerBackend):
    noisy_gate_set = {OpType.CX, OpType.H, OpType.Rz, OpType.Rz, OpType.Measure}

    def __init__(self, n_qubits: int, prob_1: float, prob_2: float, prob_ro: float):
        """AerBacked with simple depolarising and SPAM noise model.

        :param n_qubits: The number of qubits available on the backend.
        :type n_qubits: int
        :param prob_1: The depolarising noise error rates on single qubit gates.
        :type prob_1: float
        :param prob_2: The depolarising noise error rates on two qubit gates.
        :type prob_2: float
        :param prob_ro: Error rates of symmetric uncorrelated SPAM errors.
        :type prob_ro: float
        """

        noise_model = self.depolarizing_noise_model(n_qubits, prob_1, prob_2, prob_ro)
        super().__init__(noise_model=noise_model)

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
        :type n_qubits: int
        :param prob_1: The depolarising noise error rates on single qubit gates.
        :type prob_1: float
        :param prob_2: The depolarising noise error rates on two qubit gates.
        :type prob_2: float
        :param prob_ro: Error rates of symmetric uncorrelated SPAM errors.
        :type prob_ro: float
        :return: Noise model
        :rtype: noise.NoiseModel
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


class MockQuantinuumBackend(QuantinuumBackend):
    gate_set = _GATE_SET
    gate_set.add(OpType.ZZPhase)

    backend_info = BackendInfo(
        name="MockQuantinuumBackend",
        device_name="mock-quantinuum",
        version="n/a",
        architecture=FullyConnected(10, "node"),
        gate_set=gate_set,
    )

    noisy_gate_set = {OpType.CX, OpType.H, OpType.Rz, OpType.Rz, OpType.Measure}

    def __init__(self):
        super(MockQuantinuumBackend, self).__init__(device_name="H1-1SC")
        self.noisy_backend = NoisyAerBackend(
            self.backend_info.n_nodes, 0.0001, 0.001, 0.01
        )
        self.handle_cu_dict = dict()

    def process_circuit(
        self,
        circuit: Circuit,
        n_shot: int,
        valid_check: bool = True,
        **kwargs,
    ) -> ResultHandle:
        """Submit circuit to the backend for running.

        :param circuit: Circuit to process on the backend
        :type circuit: Circuit
        :param n_shot: Number of shots to run per circuit.
        :type n_shot: int
        :param valid_check: Explicitly check that all circuits satisfy all
            required predicates to run on the backend, defaults to True
        :type valid_check: bool, optional
        :return: Handles to results for each input circuit, as an interable in the same order as the circuits.
        :rtype: ResultHandle
        """

        if valid_check:
            assert self.valid_circuit(circuit)

        noisy_circuit = circuit.copy()
        cu = CompilationUnit(noisy_circuit)
        auto_rebase_pass(gateset=self.noisy_gate_set).apply(cu)
        self.noisy_backend.default_compilation_pass(optimisation_level=0).apply(cu)

        assert GateSetPredicate(self.noisy_gate_set).verify(cu.circuit)

        handle = self.noisy_backend.process_circuit(cu.circuit, n_shot)
        self.handle_cu_dict[handle] = cu

        return handle

    def process_circuits(
        self,
        circuits: List[Circuit],
        n_shots: Union[List[int], int],
        valid_check: bool = True,
        **kwargs,
    ) -> List[ResultHandle]:
        """Submit list of circuits to the backend for running.

        :param circuits: Circuits to process on the backend
        :type circuits: List[Circuit]
        :param n_shot: Number of shots to run per circuit. May be a list of
            the same length as circuits.
        :type n_shot: int
        :param valid_check: Explicitly check that all circuits satisfy all
            required predicates to run on the backend, defaults to True
        :type valid_check: bool, optional
        :return: Handles to results for each input circuit, as an interable in the same order as the circuits.
        :rtype: List[ResultHandle]
        """
        if isinstance(n_shots, int):
            return [
                self.process_circuit(circuit, n_shots, valid_check=valid_check)
                for circuit in circuits
            ]
        return [
            self.process_circuit(circuit, n_shot, valid_check=valid_check)
            for circuit, n_shot in zip(circuits, n_shots)
        ]

    def get_result(self, handle: ResultHandle, **kwargs) -> BackendResult:
        """Return a BackendResult corresponding to the handle.

        :param handle: handle to results
        :type handle: ResultHandle
        :return: Results corresponding to handle.
        :rtype: BackendResult
        """
        return self.noisy_backend.get_result(handle, *kwargs)

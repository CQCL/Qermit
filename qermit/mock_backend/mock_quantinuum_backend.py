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

from pytket.backends.backendinfo import BackendInfo
from pytket.architecture import FullyConnected  # type: ignore
from pytket.passes.auto_rebase import auto_rebase_pass
from pytket.predicates import GateSetPredicate  # type: ignore
from pytket.extensions.quantinuum import QuantinuumBackend  # type: ignore
from pytket.extensions.quantinuum.backends.quantinuum import _GATE_SET  # type: ignore
from pytket.predicates import CompilationUnit  # type: ignore
from pytket import OpType
from pytket import Circuit
from pytket.backends.resulthandle import ResultHandle
from typing import List, Union
from pytket.backends.backendresult import BackendResult
from .noisy_aer_backend import NoisyAerBackend


class MockQuantinuumBackend(QuantinuumBackend):
    """ Backend mocking some of the features of QuantinuumBackend.
    In particular the gateset and connectivity of the backend is replicated
    so that compilation behaviour is reproduced. Some noise (unrelated to
    that on the device) is also applied.
    """
    gate_set = _GATE_SET
    gate_set.add(OpType.ZZPhase)

    backend_info = BackendInfo(
        name="MockQuantinuumBackend",
        device_name="mock-quantinuum",
        version="n/a",
        architecture=FullyConnected(10, 'q'),
        gate_set=gate_set,
        n_cl_reg=100,
    )

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
        :return: Handles to results for each input circuit, as an interable
            in the same order as the circuits.
        :rtype: ResultHandle
        """

        if valid_check:
            assert self.valid_circuit(circuit)

        noisy_circuit = circuit.copy()
        cu = CompilationUnit(noisy_circuit)

        self.noisy_backend.default_compilation_pass(
            optimisation_level=0
        ).apply(cu)
        auto_rebase_pass(gateset=self.noisy_backend.noisy_gate_set).apply(cu)
        assert GateSetPredicate(
            self.noisy_backend.noisy_gate_set
        ).verify(cu.circuit)

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

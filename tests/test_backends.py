from pytket.extensions.quantinuum import QuantinuumBackend
from pytket import OpType
from pytket.backends.backendinfo import BackendInfo
from pytket.architecture import FullyConnected  # type: ignore
from pytket.passes.auto_rebase import auto_rebase_pass
from pytket.predicates import GateSetPredicate  # type: ignore
from pytket.extensions.quantinuum.backends.quantinuum import _GATE_SET
from pytket.predicates import CompilationUnit  # type: ignore
from pytket.extensions.qiskit import AerBackend
import qiskit.providers.aer.noise as noise  # type: ignore
from pytket import OpType


class NoisyAerBackend(AerBackend):

    noisy_gate_set = {
        OpType.CX,
        OpType.H,
        OpType.Rz,
        OpType.Rz,
        OpType.Measure
    }

    def __init__(self, n_qubits, prob_1, prob_2, prob_ro):

        noise_model = self.depolarizing_noise_model(
            n_qubits, prob_1, prob_2, prob_ro
        )
        super().__init__(noise_model=noise_model)

    def depolarizing_noise_model(self, n_qubits, prob_1, prob_2, prob_ro):

        noise_model = noise.NoiseModel()

        error_2 = noise.depolarizing_error(prob_2, 2)
        for edge in [[i, j] for i in range(n_qubits) for j in range(i)]:
            noise_model.add_quantum_error(error_2, ['cx'], [edge[0], edge[1]])
            noise_model.add_quantum_error(error_2, ['cx'], [edge[1], edge[0]])

        error_1 = noise.depolarizing_error(prob_1, 1)
        for node in range(n_qubits):
            noise_model.add_quantum_error(error_1, ['h', 'rx', 'rz'], [node])

        probabilities = [[1-prob_ro, prob_ro], [prob_ro, 1-prob_ro]]
        error_ro = noise.ReadoutError(probabilities)
        for i in range(n_qubits):
            noise_model.add_readout_error(error_ro, [i])

        return noise_model


class MockQuantinuumBackend(QuantinuumBackend):

    gate_set = _GATE_SET
    gate_set.add(OpType.ZZPhase)

    backend_info = BackendInfo(
        name='MockQuantinuumBackend',
        device_name='mock-quantinuum',
        version='n/a',
        architecture=FullyConnected(10),
        gate_set=gate_set
    )

    noisy_gate_set = {
        OpType.CX,
        OpType.H,
        OpType.Rz,
        OpType.Rz,
        OpType.Measure
    }

    def __init__(self):

        super(MockQuantinuumBackend, self).__init__(device_name='H1-1SC')
        self.noisy_backend = NoisyAerBackend(
            self.backend_info.n_nodes, 0.0001, 0.001, 0.01
        )
        self.handle_cu_dict = dict()

    def process_circuit(self, circuit, n_shot, valid_check=True, **kwargs):

        if valid_check:
            assert self.valid_circuit(circuit)

        noisy_circuit = circuit.copy()
        cu = CompilationUnit(noisy_circuit)
        auto_rebase_pass(gateset=self.noisy_gate_set).apply(cu)
        self.noisy_backend.default_compilation_pass(
            optimisation_level=0
        ).apply(cu)

        assert GateSetPredicate(self.noisy_gate_set).verify(cu.circuit)

        handle = self.noisy_backend.process_circuit(cu.circuit, n_shot)
        self.handle_cu_dict[handle] = cu

        return handle

    def process_circuits(self, circuits, n_shots, valid_check=True, **kwargs):
        if isinstance(n_shots, int):
            return [
                self.process_circuit(circuit, n_shots, valid_check=valid_check)
                for circuit in circuits
            ]
        return [
                self.process_circuit(circuit, n_shot, valid_check=valid_check)
                for circuit, n_shot in zip(circuits, n_shots)
            ]

    def get_result(self, handle, **kwargs):
        return self.noisy_backend.get_result(handle, *kwargs)

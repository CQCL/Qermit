from pytket.extensions.qiskit import AerBackend  # type: ignore
from collections import Counter
from pytket.backends.backendresult import BackendResult
from pytket.utils.outcomearray import OutcomeArray
import uuid


class TranspilerBackend:

    def __init__(
        self,
        transpiler,
        max_batch_size=100,
        result_dict={},
    ):

        self.transpiler = transpiler

        self.max_batch_size = max_batch_size
        self.result_dict = result_dict

        self.backend = AerBackend()

    def run_circuit(self, circuit, n_shots, **kwargs):

        handle = self.process_circuit(circuit, n_shots, **kwargs)
        return self.get_result(handle=handle)

    def process_circuit(self, circuit, n_shots, **kwargs):

        handle = uuid.uuid4()

        counts = self.get_counts(
            circuit=circuit,
            n_shots=n_shots,
            cbits=circuit.bits,
        )

        self.result_dict[handle] = BackendResult(
            counts=Counter({
                OutcomeArray.from_readouts([key]): val
                for key, val in counts.items()
            }),
            c_bits=circuit.bits,
        )

        return handle

    def get_result(self, handle):
        return self.result_dict[handle]

    def get_counts(self, circuit, n_shots, cbits=None):

        counter = Counter()

        def gen_transpiled_circuit(circuit):

            transpiled_circuit = circuit.copy()
            self.transpiler.apply(transpiled_circuit)
            self.backend.rebase_pass().apply(transpiled_circuit)
            return transpiled_circuit

        def gen_batches(circuit, n_shots):

            for _ in range(n_shots // self.max_batch_size):
                yield [
                    gen_transpiled_circuit(circuit)
                    for _ in range(self.max_batch_size)
                ]

            if n_shots % self.max_batch_size > 0:
                yield [
                    gen_transpiled_circuit(circuit)
                    for _ in range(n_shots % self.max_batch_size)
                ]

        for circuit_list in gen_batches(circuit, n_shots):
            result_list = self.backend.run_circuits(circuit_list, n_shots=1)
            counter += sum((result.get_counts(cbits=cbits)
                           for result in result_list), Counter())

        return counter

from pytket.extensions.qiskit import AerBackend  # type: ignore
from collections import Counter
from pytket.backends.backendresult import BackendResult
from pytket.utils.outcomearray import OutcomeArray
import uuid
from pytket.passes import BasePass
from typing import Dict, List, Optional, Iterator
from pytket import Circuit, Bit


class TranspilerBackend:
    """
    Provides a backend like interface for noise simulation via compiler passes.
    In particular, for each shot a new circuit is generated by applying the
    given compiler pass.

    Attributes:
        transpiler: Compiler pass to apply to simulate noise on a single
            instance of a circuit.
        max_batch_size: Shots are simulated in batches. This is the largest
            shot batch size permitted.
        result_dict: A dictionary mapping handles to results.
        backend: Backend used to simulate compiled circuits.
    """

    transpiler: BasePass
    max_batch_size: int
    result_dict: Dict[uuid.UUID, BackendResult]
    backend = AerBackend()

    def __init__(
        self,
        transpiler: BasePass,
        max_batch_size: int = 100,
        result_dict: Dict[uuid.UUID, BackendResult] = {},
    ):
        """Initialisation method.

        :param transpiler: Compiler to use during noise simulation.
        :type transpiler: BasePass
        :param max_batch_size: Size of the largest batch of shots,
            defaults to 100
        :type max_batch_size: int, optional
        :param result_dict: Results dictionary, may be used to store existing
            results within backend, defaults to {}
        :type result_dict: Dict[uuid.UUID, BackendResult], optional
        """

        self.transpiler = transpiler

        self.max_batch_size = max_batch_size
        self.result_dict = result_dict

    def run_circuit(
        self,
        circuit: Circuit,
        n_shots: int,
        **kwargs,
    ) -> BackendResult:
        """Return results of running one circuit.

        :param circuit: Circuit to run
        :type circuit: Circuit
        :param n_shots: Number of shots to be taken from circuit.
        :type n_shots: int
        :return: Result of running circuit.
        :rtype: BackendResult
        """

        handle = self.process_circuit(circuit, n_shots, **kwargs)
        return self.get_result(handle=handle)

    def process_circuit(
        self,
        circuit: Circuit,
        n_shots: int,
        **kwargs,
    ) -> uuid.UUID:
        """[summary]

        :param circuit: Submits circuit to run on noisy backend.
        :type circuit: Circuit
        :param n_shots: Number of shots to take from circuit.
        :type n_shots: int
        :return: Handle identifying results in `result_dict`.
        :rtype: uuid.UUID
        """

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

    def get_result(self, handle: uuid.UUID) -> BackendResult:
        """Retrive result from backend.

        :param handle: Handle identifying result.
        :type handle: uuid.UUID
        :return: Result corresponding to handle.
        :rtype: BackendResult
        """
        return self.result_dict[handle]

    def get_counts(
        self,
        circuit: Circuit,
        n_shots: int,
        cbits: Optional[List[Bit]] = None,
    ) -> Counter:
        """Generate shots from the given circuit.

        :param circuit: Circuit to take shots from.
        :type circuit: Circuit
        :param n_shots: Number of shots to take from circuit.
        :type n_shots: int
        :param cbits: Classical bits to return shots from,
            defaults to returning all.
        :type cbits: List[Bit], optional
        :return: Counter detailing shots from circuit.
        :rtype: Counter
        :rtype: Iterator[Counter]
        """

        counter: Counter = Counter()

        def gen_transpiled_circuit(circuit: Circuit) -> Circuit:
            """Generate compiled circuit by copying and compiling it.

            :param circuit: Circuit to be compiled.
            :type circuit: Circuit
            :return: Compiled circuit.
            :rtype: Circuit
            """

            transpiled_circuit = circuit.copy()
            self.transpiler.apply(transpiled_circuit)
            self.backend.rebase_pass().apply(transpiled_circuit)
            return transpiled_circuit

        def gen_batches(circuit: Circuit, n_shots: int) -> Iterator[List[Circuit]]:
            """Iterator generating lists of circuits of size `max_batch_size`
                until all shots have been accounted for.

            :param circuit: Circuit to batch into shots.
            :type circuit: Circuit
            :param n_shots: Number of shots to take from circuit.
            :type n_shots: int
            :return: List of compiled circuits, which is to say noisy circuits.
            :rtype: Iterator[List[Circuit]]
            """

            # Return lists of size max_batch_size containing unique
            # compiled instances of the given circuit.
            for _ in range(n_shots // self.max_batch_size):
                yield [
                    gen_transpiled_circuit(circuit)
                    for _ in range(self.max_batch_size)
                ]

            # If less than max_batch_size shots remain to be returned,
            # return what's left.
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

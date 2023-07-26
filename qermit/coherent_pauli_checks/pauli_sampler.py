import numpy.random
from quantinuum_benchmarking.direct_fidelity_estimation import Stabiliser
from abc import ABC, abstractmethod
from pytket.pauli import Pauli, QubitPauliString
from itertools import product


class PauliSampler(ABC):

    @abstractmethod
    def sample(self, **kwargs):
        pass


class DeterministicZPauliSampler(PauliSampler):

    def sample(self, qubit_list, **kwargs):
        return Stabiliser(
            Z_list=[1] * len(qubit_list),
            X_list=[0] * len(qubit_list),
            qubit_list=qubit_list,
        )


class DeterministicXPauliSampler(PauliSampler):

    def sample(self, qubit_list, **kwargs):
        return Stabiliser(
            Z_list=[0] * len(qubit_list),
            X_list=[1] * len(qubit_list),
            qubit_list=qubit_list,
        )


class RandomPauliSampler(PauliSampler):

    def __init__(self, seed=None):
        self.rng = numpy.random.default_rng(seed)

    def sample(self, qubit_list, **kwargs):
        return Stabiliser(
            Z_list=[self.rng.integers(2) for _ in qubit_list],
            X_list=[self.rng.integers(2) for _ in qubit_list],
            qubit_list=qubit_list,
        )


class OptimalPauliSampler(PauliSampler):

    def __init__(self, noise_model):
        self.noise_model = noise_model

    def sample(self, qubit_list, circ, **kwargs):

        # TODO: assert that the registers match in this case

        error_counter = self.noise_model.get_effective_pre_error_distribution(circ, **kwargs)

        smallest_commute_prob=1
        for pauli_string in product([Pauli.X , Pauli.Y, Pauli.Z, Pauli.I], repeat=circ.n_qubits):
            qubit_pauli_string = QubitPauliString(
                    qubits=circ.qubits, paulis=pauli_string
                )
            commute_prob = 0
            for error, prob in error_counter.distribution.items():
                if error.commutes_with(qubit_pauli_string):
                    commute_prob += prob
            if smallest_commute_prob>=commute_prob:
                smallest_commute_prob=commute_prob
                smallest_commute_prob_pauli=qubit_pauli_string

        print("smallest_commute_prob_pauli", smallest_commute_prob_pauli)

        return Stabiliser.from_qubit_pauli_string(smallest_commute_prob_pauli)
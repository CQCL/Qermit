import numpy.random
from quantinuum_benchmarking.direct_fidelity_estimation import Stabiliser
from abc import ABC, abstractmethod
from pytket.pauli import Pauli, QubitPauliString
from itertools import product, combinations


class PauliSampler(ABC):

    @abstractmethod
    def sample(self, **kwargs):
        pass


class DeterministicZPauliSampler(PauliSampler):

    def sample(self, qubit_list, **kwargs):
        return [Stabiliser(
            Z_list=[1] * len(qubit_list),
            X_list=[0] * len(qubit_list),
            qubit_list=qubit_list,
        )]


class DeterministicXPauliSampler(PauliSampler):

    def sample(self, qubit_list, **kwargs):
        return [Stabiliser(
            Z_list=[0] * len(qubit_list),
            X_list=[1] * len(qubit_list),
            qubit_list=qubit_list,
        )]


class RandomPauliSampler(PauliSampler):

    def __init__(self, seed=None):
        self.rng = numpy.random.default_rng(seed)

    def sample(self, qubit_list, n_checks=1, **kwargs):

        # TODO: Make sure sampling is done without replacement

        stabiliser_list = []
        while len(stabiliser_list) < n_checks:
            
            Z_list=[self.rng.integers(2) for _ in qubit_list]
            X_list=[self.rng.integers(2) for _ in qubit_list]

            # Avoids using the identity string as it commutes with all errors
            if any(Z==1 for Z in Z_list) or any(X==1 for X in X_list):
                stabiliser_list.append(
                    Stabiliser(
                        Z_list=Z_list,
                        X_list=X_list,
                        qubit_list=qubit_list,
                    )
                )
        
        return stabiliser_list


class OptimalPauliSampler(PauliSampler):

    def __init__(self, noise_model):
        self.noise_model = noise_model

    def sample(self, qubit_list, circ, n_checks=1, **kwargs):

        # TODO: assert that the registers match in this case

        # n_rand = kwargs.get("n_rand", 1000)
        # n_checks = kwargs.get("n_checks", 1)
        error_counter = self.noise_model.get_effective_pre_error_distribution(circ, **kwargs)

        smallest_commute_prob=1
        # TODO: There is probably a better way to search through this space.
        # Here are some ideas:
        #   -   It's better to prioritise checks that require fewer gates. 
        #       Here I am checking I last as it requires the fewerst gates.
        #       However note that IYY is checked after YII so IYY may be
        #       picked even though YII is lighter in the case that they
        #       have the same probability.
        #   -   It may be redundant to check Pauli.I? If a string is selected
        #       with equal probability it may be worth it though.
        #   -   Eventually we may have to select Pauli strings at random,
        #       but i'm not sure at what size that will be necessary.
        for pauli_string_list in combinations(product([Pauli.Y, Pauli.X, Pauli.Z, Pauli.I], repeat=circ.n_qubits), n_checks):
            qubit_pauli_string_list = [QubitPauliString(
                    qubits=circ.qubits, paulis=pauli_string
                ) for pauli_string in pauli_string_list]
            commute_prob = 0
            for error, prob in error_counter.distribution.items():
                if all(error.commutes_with(qubit_pauli_string) for qubit_pauli_string in qubit_pauli_string_list):
                    commute_prob += prob
            if smallest_commute_prob>=commute_prob:
                smallest_commute_prob=commute_prob
                smallest_commute_prob_pauli_list=qubit_pauli_string_list

        print("smallest_commute_prob_pauli_list", smallest_commute_prob_pauli_list)
        print("smallest_commute_prob", smallest_commute_prob)

        return [Stabiliser.from_qubit_pauli_string(smallest_commute_prob_pauli) for smallest_commute_prob_pauli in smallest_commute_prob_pauli_list]
import numpy.random
from quantinuum_benchmarking.direct_fidelity_estimation import Stabiliser  # type: ignore
from abc import ABC, abstractmethod
from pytket.pauli import Pauli, QubitPauliString  # type: ignore
from itertools import product, combinations
import warnings


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

    def __init__(self, rng=numpy.random.default_rng()):
        self.rng = rng

    def sample(
        self,
        # qubit_list,
        circ,
        n_checks=2,
        **kwargs
    ):

        print("n_checks", n_checks)

        # TODO: Make sure sampling is done without replacement

        stabiliser_list = []
        while len(stabiliser_list) < n_checks:

            # Z_list = [self.rng.integers(2) for _ in qubit_list]
            # X_list = [self.rng.integers(2) for _ in qubit_list]

            Z_list = [self.rng.integers(2) for _ in circ.qubits]
            X_list = [self.rng.integers(2) for _ in circ.qubits]

            # Avoids using the identity string as it commutes with all errors
            if any(Z == 1 for Z in Z_list) or any(X == 1 for X in X_list):
                stabiliser_list.append(
                    Stabiliser(
                        Z_list=Z_list,
                        X_list=X_list,
                        # qubit_list=qubit_list,
                        qubit_list=circ.qubits,
                    )
                )

        # print("stabiliser_list", *stabiliser_list)

        return stabiliser_list


class OptimalPauliSampler(PauliSampler):

    def __init__(self, noise_model):
        self.noise_model = noise_model

    def sample(
        self,
        circ,
        error_counter=None,
        n_checks=2,
        **kwargs,
    ):

        print("n_checks", n_checks)

        # TODO: assert that the registers match in this case

        # n_rand = kwargs.get("n_rand", 1000)
        # n_checks = kwargs.get("n_checks", 1)
        if error_counter is None:
            error_counter = self.noise_model.get_effective_pre_error_distribution(circ, **kwargs)

        total_commute_prob = 0
        total_n_pauli = 0

        # smallest_commute_prob stores the proportion of shots which will
        # still have errors.
        smallest_commute_prob = 1
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
        for pauli_string_list in combinations(
            product([Pauli.Y, Pauli.X, Pauli.Z, Pauli.I], repeat=circ.n_qubits),
            n_checks
        ):

            if tuple([Pauli.I]*circ.n_qubits) in pauli_string_list:
                continue

            qubit_pauli_string_list = [
                QubitPauliString(
                    qubits=circ.qubits,
                    paulis=pauli_string
                ) for pauli_string in pauli_string_list
            ]
            commute_prob = 0
            for error, prob in error_counter.distribution.items():
                if all(error.commutes_with(qubit_pauli_string) for qubit_pauli_string in qubit_pauli_string_list):
                    commute_prob += prob
            # print(qubit_pauli_string_list, commute_prob)
            if smallest_commute_prob >= commute_prob:
                smallest_commute_prob = commute_prob
                smallest_commute_prob_pauli_list = qubit_pauli_string_list

            total_commute_prob += commute_prob
            total_n_pauli += 1

        average_commute_prob = total_commute_prob/total_n_pauli
        print("smallest_commute_prob_pauli_list", smallest_commute_prob_pauli_list)
        print("smallest_commute_prob", smallest_commute_prob)
        print("average commute_prob", average_commute_prob)
 
        if (average_commute_prob == 0) or (abs(1-(smallest_commute_prob/average_commute_prob)) < 0.1):
            warnings.warn(
                'The smallest commute probability is close to the average. '
                + 'Random check sampling will probably work just as well.'
            )

        return [
            Stabiliser.from_qubit_pauli_string(smallest_commute_prob_pauli)
            for smallest_commute_prob_pauli in smallest_commute_prob_pauli_list
        ]

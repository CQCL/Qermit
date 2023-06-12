from .stabiliser import Stabiliser
import numpy.random

class PauliSampler:

    def sample(self, **kwargs):
        pass


class DeterministicZPauliSampler(PauliSampler):

    def __init__(self):
        pass

    def sample(self, qubit_list):
        return Stabiliser(
            Z_list=[1] * len(qubit_list),
            X_list=[0] * len(qubit_list),
            qubit_list=qubit_list,
        )

class DeterministicXPauliSampler(PauliSampler):

    def __init__(self):
        pass

    def sample(self, qubit_list):
        return Stabiliser(
            Z_list=[0] * len(qubit_list),
            X_list=[1] * len(qubit_list),
            qubit_list=qubit_list,
        )

class RandomPauliSampler(PauliSampler):

    def __init__(self, seed=None):
        self.rng = numpy.random.default_rng(seed)

    def sample(self, qubit_list, seed=None):
        numpy.random.seed(seed)
        return Stabiliser(
            Z_list=[self.rng.integers(2) for _ in qubit_list],
            X_list=[self.rng.integers(2) for _ in qubit_list],
            qubit_list=qubit_list,
        )
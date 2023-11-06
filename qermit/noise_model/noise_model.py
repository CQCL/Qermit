from __future__ import annotations
import numpy as np
from .stabiliser import Stabiliser
# from collections import Counter
import math
from pytket.circuit import OpType  # type: ignore
from matplotlib.pyplot import subplots  # type: ignore
# from pytket.pauli import Pauli  # type: ignore
# from tqdm import tqdm  # type: ignore
# from .pauli_error_transpile import PauliErrorTranspile
from typing import Dict, Tuple, List
from pytket.pauli import Pauli
from .error_sampler import ErrorSampler
from pytket import Qubit
from pytket.pauli import QubitPauliString


# TODO: For now we should make this explicitly a Pauli error distribution,
# and conduct the appropriate checks.
# TODO: Think about if we insist that the user specifies the identity 'error'
class ErrorDistribution:

    def __init__(
        self,
        distribution: Dict[Tuple[Pauli, ...], float],
        rng=np.random.default_rng(),
    ):

        if sum(distribution.values()) > 1:
            if not math.isclose(sum(distribution.values()), 1):
                raise Exception(
                    f"Probabilities sum to {sum(distribution.values())}"
                    + " but should be less than 1."
                )

        self.distribution = distribution
        self.rng = rng

    @classmethod
    def average(cls, distribution_list: List[ErrorDistribution]):

        merged_distribution = {}

        support = set(
            sum(
                [
                    list(distribution.distribution.keys())
                    for distribution in distribution_list
                ], []
            )
        )

        for error in support:
            merged_distribution[error] = sum(
                distribution.distribution.get(error, 0)
                for distribution in distribution_list
            ) / len(distribution_list)

        # total = sum(
        #     sum(distribution.distribution.values())
        #     for distribution in distribution_list
        # )
        # for error in merged_distribution:
        #     merged_distribution[error] = merged_distribution[error] / total

        return cls(distribution=merged_distribution)

    def __eq__(self, other):

        if not all(
            paulis in self.distribution.keys()
            for paulis in other.distribution.keys()
        ):
            return False
        if not all(
            paulis in other.distribution.keys()
            for paulis in self.distribution.keys()
        ):
            return False
        if not all(
            math.isclose(
                self.distribution[error],
                other.distribution[error],
                abs_tol=0.01
            )
            for error in self.distribution.keys()
        ):
            return False
        return True

    def order(self):
        self.distribution = {
            error: probability
            for error, probability
            in sorted(self.distribution.items(), key=lambda x: x[1], reverse=True)
        }

    def __str__(self):
        return ''.join(
            f"{key}:{value} \n" for key, value in self.distribution.items()
        )

    def reset_seed(self, rng):
        self.rng = rng

    def to_dict(self):
        if all(
            isinstance(error, QubitPauliString)
            for error in self.distribution.keys()
        ):
            return [
                {
                    "op_list": op_list.to_list(),
                    "noise_level": noise_level,
                    "error_type": "QubitPauliString"
                }
                for op_list, noise_level in self.distribution.items()
            ]
        else:
            return [
                {
                    "op_list": [op.value for op in op_list],
                    "noise_level": noise_level,
                    "error_type":"Tuple[Pauli]"
                }
                for op_list, noise_level in self.distribution.items()
            ]

    @classmethod
    def from_dict(cls, distribution_dict: Dict) -> ErrorDistribution:

        distribution = {}
        for noise_op in distribution_dict:
            if noise_op["error_type"] == "QubitPauliString":
                distribution[QubitPauliString.from_list(
                    noise_op["op_list"])] = noise_op['noise_level']
            else:
                distribution[
                    tuple(Pauli(op) for op in noise_op['op_list'])
                ] = noise_op['noise_level']

        return cls(distribution=distribution)

        # return cls(
        #     distribution={
        #         tuple(
        #             Pauli(op)
        #             for op in noise_op['op_list']
        #         ): noise_op['noise_level']
        #         for noise_op in distribution_dict
        #     }
        # )

    def sample(self):

        return_val = self.rng.uniform(0, 1)
        total = 0
        for error, prob in self.distribution.items():
            total += prob
            if total >= return_val:
                return error

        return None

    def plot(self):

        fig, ax = subplots()

        # TODO: There should be a neater way of doing this
        to_plot = {
            key: value
            for key, value in self.distribution.items()
            # if key != (Pauli.I, Pauli.I)
        }

        ax.bar(range(len(to_plot)), list(to_plot.values()), align='center')
        ax.set_xticks(
            ticks=range(len(to_plot)),
            labels=list(''.join(tuple(op.name for op in op_tuple))
                        for op_tuple in to_plot.keys())
        )

        return fig

    @classmethod
    def from_stabiliser_counter(cls, stabiliser_counter, **kwargs):

        total = kwargs.get('total', sum(stabiliser_counter.values()))

        error_distribution_dict = {}
        for stab, count in dict(stabiliser_counter).items():
            # Note that I am ignoring the phase here
            pauli_string, _ = stab.qubit_pauli_string
            error_distribution_dict[
                pauli_string
            ] = error_distribution_dict.get(pauli_string, 0) + count / total

        return cls(error_distribution_dict)

    def post_select(self, qubit_list):

        def string_to_pauli(pauli_string):
            qubit = Qubit(name=pauli_string[0][0], index=pauli_string[0][1][0])
            if pauli_string[1] == 'I':
                pauli = Pauli.I
            elif pauli_string[1] == 'X':
                pauli = Pauli.X
            elif pauli_string[1] == 'Y':
                pauli = Pauli.Y
            elif pauli_string[1] == 'Z':
                pauli = Pauli.Z
            else:
                raise Exception("How did you get here?")
            # match pauli_string[1]:
            #     case 'I':
            #         pauli = Pauli.I
            #     case 'X':
            #         pauli = Pauli.X
            #     case 'Y':
            #         pauli = Pauli.Y
            #     case 'Z':
            #         pauli = Pauli.Z
            #     case _:
            #         raise Exception("How did you get here?")
            return qubit, pauli

        def reduce_pauli_error(pauli_error):
            qubits_paulis = [string_to_pauli(
                pauli_string) for pauli_string in pauli_error.to_list()]
            qubits, paulis = zip(
                *[(qubits, paulis) for qubits, paulis in qubits_paulis if qubits not in qubit_list])
            return QubitPauliString(
                qubits=qubits,
                paulis=paulis,
            )

        distribution = {}
        for pauli_error, probability in self.distribution.items():
            pauli_error_stabiliser = Stabiliser.from_qubit_pauli_string(
                pauli_error)
            if not pauli_error_stabiliser.is_measureable(qubit_list):
                distribution[reduce_pauli_error(pauli_error)] = distribution.get(
                    reduce_pauli_error(pauli_error), 0) + probability
        return ErrorDistribution(distribution=distribution)


class NoiseModel:

    def __init__(self, noise_model: Dict[OpType, ErrorDistribution]):

        self.noise_model = noise_model
        # self.transpiiler = PauliErrorTranspile(self)
        self.error_sampler = ErrorSampler(self)

    def reset_seed(self, rng):
        for distribution in self.noise_model.values():
            distribution.reset_seed(rng=rng)

    def plot(self):

        fig_list = []
        for noisy_gate, distribution in self.noise_model.items():
            fig = distribution.plot()
            ax = fig.axes[0]
            ax.set_title(noisy_gate.name)
            fig_list.append(fig)

        return fig_list

    def __eq__(self, other):

        if not (
            sorted(self.noise_model.keys())
            == sorted(other.noise_model.keys())
        ):
            return False
        if not all(
            self.noise_model[op] == other.noise_model[op]
            for op in self.noise_model.keys()
        ):
            return False

        return True

    def to_dict(self):
        return {
            op.name: distribution.to_dict()
            for op, distribution in self.noise_model.items()
        }

    @classmethod
    def from_dict(cls, noise_model_dict: Dict) -> NoiseModel:
        return cls(
            noise_model={
                OpType.from_name(op): ErrorDistribution.from_dict(
                    error_distribution
                )
                for op, error_distribution in noise_model_dict.items()
            }
        )

    @property
    def noisy_gates(self):
        return list(self.noise_model.keys())

    def get_error_distribution(self, optype):
        return self.noise_model[optype]

    def get_effective_pre_error_distribution(
        self,
        cliff_circ,
        n_rand=1000,
        **kwargs,
    ) -> ErrorDistribution:

        print("n_rand", n_rand)

        # error_counter = Counter()
        # is_identity_count = 0

        # # There is some time wasted here, if for example there is no error in
        # # back_propagate_random_error. There may be a saving to be made here
        # # if there errors are sampled before the back propagation occurs?
        # for _ in tqdm(range(n_rand), desc='Sampling effective error channel'):
        #     # stabiliser = self.back_propagate_random_error(cliff_circ)
        #     stabiliser = self.error_sampler.random_back_propagate(cliff_circ)

        #     if stabiliser.is_identity():
        #         is_identity_count+=1
        #         continue

        #     error_counter.update([stabiliser])

        error_counter = self.error_sampler.counter_propagate(
            cliff_circ=cliff_circ,
            n_counts=n_rand,
            direction='backward',
        )

        # error_distribution_dict = {}
        # for stab, count in dict(error_counter).items():
        #     # Note that I am ignoring the phase here which is safe
        #     pauli_string, _ = stab.qubit_pauli_string
        #     error_distribution_dict[
        #         pauli_string
        #     ] = error_distribution_dict.get(pauli_string, 0) + count/n_rand

        # error_distribution = ErrorDistribution(error_distribution_dict)

        error_distribution = ErrorDistribution.from_stabiliser_counter(
            error_counter, total=n_rand)

        # print("error_distribution \n", error_distribution)
        # print("is_identity_count", is_identity_count)

        return error_distribution

from __future__ import annotations
import numpy as np
from .stabiliser import Stabiliser
from collections import Counter
import math
from pytket.circuit import OpType  # type: ignore
from matplotlib.pyplot import subplots  # type: ignore
from typing import Dict, Tuple, List, Union, cast
from pytket.pauli import Pauli
from pytket import Qubit
from pytket.pauli import QubitPauliString
from numpy.random import Generator


class ErrorDistribution:
    """
    Model of a Pauli error channel. Contains utilities to analyse and
    sample from distributions of errors.

    Attributes:
        distribution: Dictionary mapping a string of Pauli errors
            to the probability that they occur.
        rng: Randomness generator.
    """

    distribution: Dict[Tuple[Pauli, ...], float]
    rng: Generator

    def __init__(
        self,
        distribution: Dict[Tuple[Pauli, ...], float],
        rng: Generator = np.random.default_rng(),
    ):
        """Initialisation method.

        :param distribution: Dictionary mapping a string of Pauli errors
            to the probability that they occur.
        :type distribution: Dict[Tuple[Pauli, ...], float]
        :param rng: Randomness generator, defaults to np.random.default_rng()
        :type rng: Generator, optional
        :raises Exception: Raised if error probabilities sum to greater than 1.
        """

        if sum(distribution.values()) > 1:
            if not math.isclose(sum(distribution.values()), 1):
                raise Exception(
                    f"Probabilities sum to {sum(distribution.values())}"
                    + " but should be less than or equal to 1."
                )

        self.distribution = distribution
        self.rng = rng

    def __eq__(self, other: object) -> bool:
        """Check equality of two instances of ErrorDistribution by ensuring
        that all keys in distribution match, and that the probabilities are
        close for each value.

        :param other: Instance of ErrorDistribution to be compared against.
        :type other: object
        :return: True if two instances are equal, false otherwise.
        :rtype: bool
        """

        if not isinstance(other, ErrorDistribution):
            return False

        # Check all pauli error in this distribution are in the other.
        if not all(
            paulis in self.distribution.keys()
            for paulis in other.distribution.keys()
        ):
            return False
        # Check all pauli errors in the other distribution are in this one.
        if not all(
            paulis in other.distribution.keys()
            for paulis in self.distribution.keys()
        ):
            return False
        # Check all probabilities are close.
        if not all(
            math.isclose(
                self.distribution[error],
                other.distribution[error],
                abs_tol=0.01
            )
            for error in self.distribution.keys()
        ):
            return False

        # Otherwise they are equal.
        return True

    def __str__(self) -> str:
        """Generates string representation of error distribution.

        :return: String representation of error distribution.
        :rtype: str
        """
        return ''.join(
            f"{key}:{value} \n" for key, value in self.distribution.items()
        )

    @classmethod
    def mixture(cls, distribution_list: List[ErrorDistribution]) -> ErrorDistribution:
        """Generates the distribution corresponding to the mixture of a
        list of distributions.

        :param distribution_list: List of instances of ErrorDistribution.
        :type distribution_list: List[ErrorDistribution]
        :return: Mixture distribution.
        :rtype: ErrorDistribution
        """

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

        return cls(distribution=merged_distribution)

    def order(self, reverse: bool = True):
        """Reorders the distribution dictionary based on probabilities.

        :param reverse: Order from hight to low, defaults to True
        :type reverse: bool, optional
        """
        self.distribution = {
            error: probability
            for error, probability
            in sorted(self.distribution.items(), key=lambda x: x[1], reverse=reverse)
        }

    def reset_seed(self, rng: Generator):
        """Reset randomness generator.

        :param rng: Randomness generator.
        :type rng: Generator
        """
        self.rng = rng

    def to_dict(self) -> List[Dict[str, Union[List[int], float]]]:
        """Produces json serialisable representation of ErrorDistribution.

        :return: Json serialisable representation of ErrorDistribution.
        :rtype: List[Dict[str, Union[List[int], float]]]
        """
        return [
            {
                "op_list": [op.value for op in op_list],
                "noise_level": noise_level,
            }
            for op_list, noise_level in self.distribution.items()
        ]

    @classmethod
    def from_dict(
        cls,
        distribution_dict: List[Dict[str, Union[List[int], float]]]
    ) -> ErrorDistribution:
        """Generates ErrorDistribution from json serialisable representation.

        :param distribution_dict: List of dictionaries, each of which map
            a property of the distribution to its value.
        :type distribution_dict: List[Dict[str, Union[List[int], float]]]
        :return: ErrorDistribution created from serialised representation.
        :rtype: ErrorDistribution
        """

        distribution = {}
        for noise_op in distribution_dict:
            distribution[
                tuple(Pauli(op) for op in cast(List[int], noise_op['op_list']))
            ] = cast(float, noise_op['noise_level'])

        return cls(distribution=distribution)

    def sample(self) -> Union[Tuple[Pauli, ...], None]:
        """Draw sample from distribution.

        :return: Either one of the pauli strings in the support of the
            distribution, or None. None can be returned if the total proability
            of the distribution not 1, and should be interpreted as the
            the unspecified support.
        :rtype: Union[Tuple[Pauli, ...], None]
        """

        return_val = self.rng.uniform(0, 1)
        total = 0.0
        for error, prob in self.distribution.items():
            total += prob
            if total >= return_val:
                return error

        return None

    def plot(self):
        """
        Generates plot of distribution.
        """

        fig, ax = subplots()

        to_plot = {
            key: value
            for key, value in self.distribution.items()
        }

        ax.bar(range(len(to_plot)), list(to_plot.values()), align='center')
        ax.set_xticks(
            ticks=range(len(to_plot)),
            labels=list(''.join(tuple(op.name for op in op_tuple))
                        for op_tuple in to_plot.keys())
        )

        return fig


class LogicalErrorDistribution:
    """
    Class for managing distributions of logical errors from a noisy
    simulation of a circuit. This differs from ErrorDistribution in that
    the errors are pauli strings rather than tuples of Paulis. That is to
    say that the errors are fixed to qubits.

    Attributes:
        stabiliser_counter: Counts number of stabilisers in error distribution
    """

    stabiliser_counter: Counter[Stabiliser]

    def __init__(self, stabiliser_counter: Counter[Stabiliser], **kwargs):
        """Initialisation method. Stores stabiliser counter.

        :param stabiliser_counter: Counter of Stabilisers.
        :type stabiliser_counter: Counter[Stabiliser]

        :key total: The total number of shots taken when measuring the
            errors. By default this will be taken to be the total
            number of errors.
        """

        self.stabiliser_counter = stabiliser_counter
        self.total = kwargs.get('total', sum(self.stabiliser_counter.values()))

    @property
    def distribution(self) -> Dict[QubitPauliString, float]:
        """Probability distribution equivalent to counts distribution.

        :return: Dictionary mapping QubitPauliString to probability that
            that error occurs.
        :rtype: Dict[QubitPauliString, float]
        """

        distribution: Dict[QubitPauliString, float] = {}
        for stab, count in dict(self.stabiliser_counter).items():
            # Note that I am ignoring the phase here
            pauli_string, _ = stab.qubit_pauli_string
            distribution[
                pauli_string
            ] = distribution.get(pauli_string, 0) + count / self.total

        return distribution

    def post_select(self, qubit_list: List[Qubit]) -> LogicalErrorDistribution:

        def reduce_pauli_error(pauli_error_stabiliser):

            return Stabiliser(
                Z_list=[Z for qubit, Z in pauli_error_stabiliser.Z_list.items() if qubit not in qubit_list],
                X_list=[X for qubit, X in pauli_error_stabiliser.X_list.items() if qubit not in qubit_list],
                qubit_list=[qubit for qubit in pauli_error_stabiliser.qubit_list if qubit not in qubit_list],
                phase=pauli_error_stabiliser.phase
            )

        # the number of error free shots.
        total = self.total - sum(self.stabiliser_counter.values())

        distribution: Counter[Stabiliser] = Counter()
        for pauli_error_stabiliser, count in self.stabiliser_counter.items():
            if not pauli_error_stabiliser.is_measureable(qubit_list):
                distribution[reduce_pauli_error(pauli_error_stabiliser)] += count
                total += count

        return LogicalErrorDistribution(stabiliser_counter=distribution, total=total)


class NoiseModel:

    def __init__(self, noise_model: Dict[OpType, ErrorDistribution]):

        self.noise_model = noise_model

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
    ) -> LogicalErrorDistribution:

        error_counter = self.counter_propagate(
            cliff_circ=cliff_circ,
            n_counts=n_rand,
            direction='backward',
        )

        return LogicalErrorDistribution(error_counter, total=n_rand)

    def counter_propagate(self, cliff_circ, n_counts=1000, **kwargs):

        error_counter = Counter()

        # There is some time wasted here, if for example there is no error in
        # back_propagate_random_error. There may be a saving to be made here
        # if there errors are sampled before the back propagation occurs?
        for _ in range(n_counts):
            stabiliser = self.random_propagate(cliff_circ, **kwargs)

            if not stabiliser.is_identity():
                error_counter.update([stabiliser])

        return error_counter

    def random_propagate(self, cliff_circ, direction='backward'):

        qubit_list = cliff_circ.qubits
        stabiliser = Stabiliser(
            Z_list=[0] * len(qubit_list),
            X_list=[0] * len(qubit_list),
            qubit_list=qubit_list,
        )

        if direction == 'backward':
            command_list = list(reversed(cliff_circ.get_commands()))
        elif direction == 'forward':
            command_list = cliff_circ.get_commands()
        else:
            raise Exception(
                f"Direction must be 'backward' or 'forward'. Is {direction}"
            )

        for command in command_list:

            if command.op.type in [OpType.Measure, OpType.Barrier]:
                continue

            if direction == 'forward':

                stabiliser.apply_gate(
                    op_type=command.op.type,
                    qubits=command.args,
                    params=command.op.params,
                )

            if command.op.type in self.noisy_gates:

                error_distribution = self.get_error_distribution(
                    optype=command.op.type
                )
                error = error_distribution.sample()

                if error is not None:
                    for pauli, qubit in zip(error, command.args):
                        if direction == 'backward':
                            stabiliser.pre_apply_pauli(
                                pauli=pauli, qubit=qubit)
                        elif direction == 'forward':
                            stabiliser.post_apply_pauli(
                                pauli=pauli, qubit=qubit)
                        else:
                            raise Exception(
                                "Direction must be 'backward' or 'forward'. "
                                + f"Is {direction}"
                            )

            if direction == 'backward':

                # Note that here we wish to pull the pauli back through the gate,
                # which has the same effect on the pauli as pushing through the
                # dagger.
                stabiliser.apply_gate(
                    op_type=command.op.dagger.type,
                    qubits=command.args,
                    params=command.op.dagger.params,
                )

        return stabiliser

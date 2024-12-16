from __future__ import annotations

import math
from collections import Counter
from enum import Enum
from itertools import product
from typing import Dict, List, Tuple, Union, cast

import numpy as np
from matplotlib.pyplot import subplots
from numpy.random import Generator
from numpy.typing import NDArray
from pytket import Circuit, Qubit
from pytket.circuit import OpType
from pytket.pauli import Pauli, QubitPauliString, QubitPauliTensor
from scipy.linalg import fractional_matrix_power  # type: ignore

from .qermit_pauli import QermitPauli

Direction = Enum("Direction", ["forward", "backward"])


class ErrorDistribution:
    """
    Model of a Pauli error channel. Contains utilities to analyse and
    sample from distributions of errors.

    Attributes:
        distribution: Dictionary mapping a string of Pauli errors to the
        probability that they occur.
        rng: Randomness generator.
    """

    distribution: Dict[Tuple[Pauli, ...], float]
    rng: Generator

    def __init__(
        self,
        distribution: Dict[Tuple[Pauli, ...], float],
        rng: Generator = np.random.default_rng(),
    ):
        """
        :param distribution: Dictionary mapping a string of Pauli errors
            to the probability that they occur.
        :param rng: Randomness generator, defaults to np.random.default_rng()
        :raises Exception: Raised if error probabilities sum to greater than 1.
        """

        if sum(distribution.values()) > 1:
            if not math.isclose(sum(distribution.values()), 1):
                raise Exception(
                    f"Probabilities sum to {sum(distribution.values())}"
                    + " but should be less than or equal to 1."
                )

        # If the given distribution is empty then no
        # noise will be acted.
        if distribution == {}:
            pass
        # If it it not empty then we check that the number
        # of qubits in each of the errors match.
        else:
            n_qubits = len(list(distribution.keys())[0])
            if not all(len(error) == n_qubits for error in distribution.keys()):
                raise Exception("Errors must all act on the same number of qubits.")

        self.distribution = distribution
        self.rng = rng

    @property
    def identity_error_rate(self) -> float:
        """The rate at which no error occurs.

        :return: Rate at which no error occurs.
            Calculated as 1 minus the total error rate of
            error in this distribution.
        """
        return 1 - sum(self.distribution.values())

    def to_ptm(self) -> Tuple[NDArray, Dict[Tuple[Pauli, ...], int]]:
        """Convert error distribution to Pauli Transfer Matrix (PTM) form.

        :return: PTM of error distribution and Pauli index dictionary.
            The Pauli index dictionary maps Pauli errors to their
            index in the PTM
        """

        # Initialise an empty PTM and index dictionary
        # of the appropriate size.
        ptm = np.zeros((4**self.n_qubits, 4**self.n_qubits))
        pauli_index = {
            pauli: index
            for index, pauli in enumerate(
                product({Pauli.I, Pauli.X, Pauli.Y, Pauli.Z}, repeat=self.n_qubits)
            )
        }

        # For each pauli, calculate the corresponding
        # PTM entry as a sum pf error weights multiplied by +/-1
        # Depending on commutation relations.
        for pauli_tuple, index in pauli_index.items():
            pauli = QermitPauli(
                QubitPauliTensor(
                    string=QubitPauliString(
                        paulis=list(pauli_tuple),
                        qubits=[Qubit(i) for i in range(self.n_qubits)],
                    ),
                    coeff=1,
                )
            )

            # Can add the identity error rate.
            # This will not come up in the following for loop
            # as the error distribution does not save
            # the rate at which no errors occur.
            ptm[index][index] += self.identity_error_rate

            for error, error_rate in self.distribution.items():
                error_pauli = QermitPauli(
                    QubitPauliTensor(
                        string=QubitPauliString(
                            paulis=list(error),
                            qubits=[Qubit(i) for i in range(self.n_qubits)],
                        ),
                        coeff=1,
                    )
                )
                commute_coeff = (
                    1
                    if pauli.qubit_pauli_tensor.commutes_with(
                        error_pauli.qubit_pauli_tensor
                    )
                    else -1
                )
                ptm[index][index] += error_rate * commute_coeff

        # Some checks that the form of the PTM is correct.
        identity = tuple(Pauli.I for _ in range(self.n_qubits))
        if not abs(ptm[pauli_index[identity]][pauli_index[identity]] - 1.0) < 10 ** (
            -6
        ):
            raise Exception(
                "The identity entry of the PTM is incorrect. "
                + "This is a fault in Qermit. "
                + "Please report this as an issue."
            )

        if not self == ErrorDistribution.from_ptm(ptm=ptm, pauli_index=pauli_index):
            raise Exception(
                "From PTM does not match to PTM. "
                + "This is a fault in Qermit. "
                + "Please report this as an issue."
            )

        return ptm, pauli_index

    @classmethod
    def from_ptm(
        cls, ptm: NDArray, pauli_index: Dict[Tuple[Pauli, ...], int]
    ) -> ErrorDistribution:
        """Convert a Pauli Transfer Matrix (PTM) to an error distribution.

        :param ptm: Pauli Transfer Matrix to convert. Should be a 4^n by 4^n matrix
            where n is the number of qubits.
        :param pauli_index: A dictionary mapping Pauli errors to
            their index in the PTM.
        :return: The converted error distribution.
        """

        if ptm.ndim != 2:
            raise Exception(
                f"This given matrix is not has dimension {ptm.ndim} "
                + "but should have dimension 2."
            )

        if ptm.shape[0] != ptm.shape[1]:
            raise Exception(
                "The dimensions of the given PTM are "
                + f"{ptm.shape[0]} and {ptm.shape[1]} "
                + "but they should match."
            )

        n_qubit = math.log(ptm.shape[0], 4)
        if n_qubit % 1 != 0.0:
            raise Exception(
                "The given PTM should have a dimension of the form 4^n "
                + "where n is the number of qubits."
            )

        if not np.array_equal(ptm, np.diag(np.diag(ptm))):
            raise Exception("The given PTM is not diagonal as it should be.")

        # calculate the error rates by solving simultaneous
        # linear equations. In particular the matrix to invert
        # is the matrix of commutation values.
        commutation_matrix = np.zeros(ptm.shape)
        for pauli_one_tuple, index_one in pauli_index.items():
            pauli_one = QermitPauli(
                QubitPauliTensor(
                    string=QubitPauliString(
                        paulis=list(pauli_one_tuple),
                        qubits=[Qubit(i) for i in range(len(pauli_one_tuple))],
                    ),
                    coeff=1,
                )
            )
            for pauli_two_tuple, index_two in pauli_index.items():
                pauli_two = QermitPauli(
                    QubitPauliTensor(
                        string=QubitPauliString(
                            paulis=list(pauli_two_tuple),
                            qubits=[Qubit(i) for i in range(len(pauli_two_tuple))],
                        ),
                        coeff=1,
                    )
                )

                commutation_matrix[index_one][index_two] = (
                    1
                    if pauli_one.qubit_pauli_tensor.commutes_with(
                        pauli_two.qubit_pauli_tensor
                    )
                    else -1
                )

        error_rate_list = np.matmul(ptm.diagonal(), np.linalg.inv(commutation_matrix))
        distribution = {
            error: error_rate_list[index]
            for error, index in pauli_index.items()
            if (error_rate_list[index] > 10 ** (-6))
            and error != tuple(Pauli.I for _ in range(int(n_qubit)))
        }
        return cls(distribution=distribution)

    @property
    def n_qubits(self) -> int:
        """The number of qubits this error distribution acts on."""
        return len(list(self.distribution.keys())[0])

    def __eq__(self, other: object) -> bool:
        """Check equality of two instances of ErrorDistribution by ensuring
        that all keys in distribution match, and that the probabilities are
        close for each value.

        :param other: Instance of ErrorDistribution to be compared against.
        :return: True if two instances are equal, false otherwise.
        """

        if not isinstance(other, ErrorDistribution):
            return False

        # Check all pauli error in this distributions are the same.
        if set(self.distribution.keys()) != set(other.distribution.keys()):
            return False

        # Check all probabilities are close.
        if not all(
            math.isclose(
                self.distribution[error], other.distribution[error], abs_tol=0.01
            )
            for error in self.distribution.keys()
        ):
            return False

        # Otherwise they are equal.
        return True

    def __str__(self) -> str:
        """Generates string representation of error distribution.

        :return: String representation of error distribution.
        """
        return "".join(f"{key}:{value} \n" for key, value in self.distribution.items())

    @classmethod
    def mixture(cls, distribution_list: List[ErrorDistribution]) -> ErrorDistribution:
        """Generates the distribution corresponding to the mixture of a
        list of distributions.

        :param distribution_list: List of instances of ErrorDistribution.
        :return: Mixture distribution.
        """

        return cls(
            distribution={
                error: sum(
                    distribution.distribution.get(error, 0)
                    for distribution in distribution_list
                )
                / len(distribution_list)
                for error in set(
                    error
                    for distribution in distribution_list
                    for error in distribution.distribution
                )
            }
        )

    def order(self, reverse: bool = True):
        """Reorders the distribution dictionary based on probabilities.

        :param reverse: Order from high to low, defaults to True
        """
        self.distribution = {
            error: probability
            for error, probability in sorted(
                self.distribution.items(), key=lambda x: x[1], reverse=reverse
            )
        }

    def reset_rng(self, rng: Generator):
        """Reset randomness generator.

        :param rng: Randomness generator.
        """
        self.rng = rng

    def to_dict(self) -> List[Dict[str, Union[List[int], float]]]:
        """Produces json serialisable representation of ErrorDistribution.

        :return: Json serialisable representation of ErrorDistribution.
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
        cls, distribution_dict: List[Dict[str, Union[List[int], float]]]
    ) -> ErrorDistribution:
        """Generates ErrorDistribution from json serialisable representation.

        :param distribution_dict: List of dictionaries, each of which map
            a property of the distribution to its value.
        :return: ErrorDistribution created from serialised representation.
        """

        return cls(
            distribution={
                tuple(Pauli(op) for op in cast(List[int], noise_op["op_list"])): cast(
                    float, noise_op["noise_level"]
                )
                for noise_op in distribution_dict
            }
        )

    def sample(self) -> Union[Tuple[Pauli, ...], None]:
        """Draw sample from distribution.

        :return: Either one of the pauli strings in the support of the
            distribution, or None. None can be returned if the total proability
            of the distribution not 1, and should be interpreted as the
            the unspecified support.
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

        to_plot = {key: value for key, value in self.distribution.items()}

        ax.bar(range(len(to_plot)), list(to_plot.values()), align="center")
        ax.set_xticks(
            ticks=range(len(to_plot)),
            labels=list(
                "".join(tuple(op.name for op in op_tuple))
                for op_tuple in to_plot.keys()
            ),
        )

        return fig

    def scale(self, scaling_factor: float) -> ErrorDistribution:
        """Scale the error rates of this error distribution.
        This is done by converting the error distribution to a PTM,
        scaling that matrix appropriately, and converting back to a
        new error distribution.

        :param scaling_factor: The factor by which the noise should be scaled.
        :return: A new error distribution with the noise scaled.
        """

        ptm, pauli_index = self.to_ptm()
        scaled_ptm = fractional_matrix_power(ptm, scaling_factor)
        return ErrorDistribution.from_ptm(ptm=scaled_ptm, pauli_index=pauli_index)


class LogicalErrorDistribution:
    """
    Class for managing distributions of logical errors from a noisy
    simulation of a circuit. This differs from ErrorDistribution in that
    the errors are pauli strings rather than tuples of Paulis. That is to
    say that the errors are fixed to qubits.

    Attributes:
        pauli_error_counter: Counts number of pauli errors in distribution
    """

    pauli_error_counter: Counter[QermitPauli]

    def __init__(self, pauli_error_counter: Counter[QermitPauli], **kwargs):
        """Initialisation method. Stores pauli error counter.

        :param pauli_error_counter: Counter of pauli errors.

        :key total: The total number of shots taken when measuring the
            errors. By default this will be taken to be the total
            number of errors.
        """

        self.pauli_error_counter = pauli_error_counter
        self.total = kwargs.get("total", sum(self.pauli_error_counter.values()))

    @property
    def distribution(self) -> Dict[QubitPauliString, float]:
        """Probability distribution equivalent to counts distribution.

        :return: Dictionary mapping QubitPauliString to probability that
            that error occurs.
        """

        distribution: Dict[QubitPauliString, float] = {}
        for stab, count in dict(self.pauli_error_counter).items():
            # Note that the phase is ignored here
            pauli_string = stab.qubit_pauli_tensor.string
            distribution[pauli_string] = (
                distribution.get(pauli_string, 0) + count / self.total
            )

        return distribution

    def post_select(self, qubit_list: List[Qubit]) -> LogicalErrorDistribution:
        """Post select based on the given qubits. In particular remove the
        the given qubits, and the shots with measurable errors on those qubits.

        :param qubit_list: List of qubits to be post selected on.
        :return: New LogicalErrorDistribution with given qubits removed,
            and those shots where there are measurable errors on those
            qubits removed.
        """

        # the number of error free shots.
        total = self.total - sum(self.pauli_error_counter.values())

        distribution: Counter[QermitPauli] = Counter()
        for pauli_error, count in self.pauli_error_counter.items():
            if not pauli_error.is_measureable(qubit_list):
                distribution[pauli_error.reduce_qubits(qubit_list)] += count
                total += count

        return LogicalErrorDistribution(
            pauli_error_counter=distribution,
            total=total,
        )


class NoiseModel:
    """
    Module for managing and executing a circuit noise model. In particular
    error models are assigned to each gate, and logical errors from
    circuits can be sampled.

    Attributes:
        noise_model: Mapping from gates to the error model which corresponds
            to that gate.
    """

    noise_model: Dict[OpType, ErrorDistribution]

    def __init__(self, noise_model: Dict[OpType, ErrorDistribution]):
        """
        :param noise_model: Map from gates to their error models.
        """

        self.noise_model = noise_model

    def scale(self, scaling_factor: float) -> NoiseModel:
        """Generate new error model where all error rates have been scaled by
        the given scaling factor.

        :param scaling_factor: Factor by which to scale the error rates.
        :return: New noise model with scaled error rates.
        """
        return NoiseModel(
            noise_model={
                op_type: error_distribution.scale(scaling_factor=scaling_factor)
                for op_type, error_distribution in self.noise_model.items()
            }
        )

    def reset_rng(self, rng: Generator):
        """Reset randomness generator.

        :param rng: Randomness generator to be reset to.
        """
        for distribution in self.noise_model.values():
            distribution.reset_rng(rng=rng)

    def plot(self):
        """Generates plot of noise model."""

        fig_list = []
        for noisy_gate, distribution in self.noise_model.items():
            fig = distribution.plot()
            ax = fig.axes[0]
            ax.set_title(noisy_gate.name)
            fig_list.append(fig)

        return fig_list

    def __eq__(self, other: object) -> bool:
        """Checks equality by checking all gates in the two noise models match,
        and that the noise models of each gate match.

        :param other: Noise model to be compared against.
        :return: True if equivalent, false otherwise.
        """

        if not isinstance(other, NoiseModel):
            return False

        if not (sorted(self.noise_model.keys()) == sorted(other.noise_model.keys())):
            return False
        if not all(
            self.noise_model[op] == other.noise_model[op]
            for op in self.noise_model.keys()
        ):
            return False

        return True

    def to_dict(self) -> Dict[str, List[Dict[str, Union[List[int], float]]]]:
        """Json serialisable object representing noise model.

        :return: Json serialisable object representing noise model.
        """
        return {
            op.name: distribution.to_dict()
            for op, distribution in self.noise_model.items()
        }

    @classmethod
    def from_dict(
        cls, noise_model_dict: Dict[str, List[Dict[str, Union[List[int], float]]]]
    ) -> NoiseModel:
        """Convert JSON serialised version of noise model back to an instance
        of NoiseModel.

        :param noise_model_dict: JSON serialised version of NoiseModel
        :return: Instance of noise model corresponding to JSON serialised
            version.
        """
        return cls(
            noise_model={
                OpType.from_name(op): ErrorDistribution.from_dict(error_distribution)
                for op, error_distribution in noise_model_dict.items()
            }
        )

    @property
    def noisy_gates(self) -> List[OpType]:
        """List of OpTypes with noise.

        :return: List of OpTypes with noise.
        """
        return list(self.noise_model.keys())

    def get_error_distribution(self, optype: OpType) -> ErrorDistribution:
        """Recovers error model corresponding to particular OpType.

        :param optype: OpType for which noise model should be retrieved.
        :return: Error model corresponding to particular OpType.
        """
        return self.noise_model[optype]

    def get_effective_pre_error_distribution(
        self,
        cliff_circ: Circuit,
        n_rand: int = 1000,
        **kwargs,
    ) -> LogicalErrorDistribution:
        """Retrieve the effective noise model of a given circuit. This is to
        say, repeatedly generate circuits with coherent noise added at random.
        Push all errors to the front of the circuit. Return a counter of the
        errors which have been pushed to the front.

        :param cliff_circ: Circuit to be simulated. This should be a Clifford
            circuit.
        :param n_rand: Number of random circuit instances, defaults to 1000
        :return: Resulting distribution of errors.
        """

        error_counter = self.counter_propagate(
            cliff_circ=cliff_circ,
            n_counts=n_rand,
            direction=Direction.backward,
        )

        return LogicalErrorDistribution(error_counter, total=n_rand)

    def counter_propagate(
        self, cliff_circ: Circuit, n_counts: int, **kwargs
    ) -> Counter[QermitPauli]:
        """Generate random noisy instances of the given circuit and propagate
        the noise to create a counter of logical errors. Note that
        kwargs are passed onto `random_propagate`.

        :param cliff_circ: Circuit to be simulated. This should be a Clifford
            circuit.
        :param n_counts: Number of random instances.
        :return: Counter of logical errors.
        """

        error_counter: Counter[QermitPauli] = Counter()

        # TODO: There is some time wasted here, if for example there is
        # no error in back_propagate_random_error. There may be a saving to
        # be made here if there errors are sampled before the back
        # propagation occurs?
        for _ in range(n_counts):
            pauli_error = self.random_propagate(cliff_circ, **kwargs)

            # Check if the error is the identity.
            if pauli_error.qubit_pauli_tensor.string != QubitPauliString():
                error_counter.update([pauli_error])

        return error_counter

    def random_propagate(
        self,
        cliff_circ: Circuit,
        direction: Direction = Direction.backward,
    ) -> QermitPauli:
        """Generate a random noisy instance of the given circuit and
        propagate the noise forward or backward to recover the logical error.

        :param cliff_circ: Circuit to be simulated. This should be a Clifford
            circuit.
        :param direction: Direction in which noise should be propagated,
            defaults to 'backward'
        :raises Exception: Raised if direction is invalid.
        :return: Resulting logical error.
        """
        pauli_error = QermitPauli(
            QubitPauliTensor(
                string=QubitPauliString(
                    map={qubit: Pauli.I for qubit in cliff_circ.qubits}
                ),
                coeff=1,
            )
        )

        # Commands are ordered in reverse or original order depending on which
        # way the noise is being pushed.
        if direction == Direction.backward:
            command_list = list(reversed(cliff_circ.get_commands()))
        elif direction == Direction.forward:
            command_list = cliff_circ.get_commands()
        else:
            raise Exception(
                "Direction must be Direction.backward or Direction.forward."
            )

        # For each command in the circuit, add an error as appropriate, and
        # push the total error through the command.
        for command in command_list:
            if command.op.type in [OpType.Measure, OpType.Barrier]:
                continue

            if direction == Direction.forward:
                # Apply gate to total error.
                pauli_error.apply_gate(
                    op=command.op,
                    qubits=cast(List[Qubit], command.args),
                )
            # Add noise operation if appropriate.
            if command.op.type in self.noisy_gates:
                error_distribution = self.get_error_distribution(optype=command.op.type)
                error = error_distribution.sample()

                if error is not None:
                    for pauli, qubit in zip(error, command.args):
                        if direction == Direction.backward:
                            pauli_error.pre_apply_pauli(
                                pauli=pauli, qubit=cast(Qubit, qubit)
                            )
                        elif direction == Direction.forward:
                            pauli_error.post_apply_pauli(
                                pauli=pauli, qubit=cast(Qubit, qubit)
                            )
                        else:
                            raise Exception(
                                "Direction must be Direction.backward or Direction.forward. "
                            )

            if direction == Direction.backward:
                # Note that here we wish to pull the pauli back through the gate,
                # which has the same effect on the pauli as pushing through the
                # dagger.
                pauli_error.apply_gate(
                    op=command.op.dagger,
                    qubits=cast(List[Qubit], command.args),
                )

        return pauli_error

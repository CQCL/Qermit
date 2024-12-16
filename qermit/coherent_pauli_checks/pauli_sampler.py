import warnings
from abc import ABC, abstractmethod
from itertools import combinations, product
from typing import List, Tuple, cast

import numpy.random
from numpy.random import Generator
from pytket import Circuit
from pytket._tket.unit_id import UnitID
from pytket.circuit import Bit, CircBox, Command, OpType, Qubit
from pytket.pauli import Pauli, QubitPauliString, QubitPauliTensor

from qermit.noise_model.noise_model import NoiseModel
from qermit.noise_model.qermit_pauli import QermitPauli


class PauliSampler(ABC):
    """Abstract base class for Pauli samplers. Pauli samples should sample
    Paulis to be used a checks.
    """

    @abstractmethod
    def sample(self, circ: Circuit) -> List[QermitPauli]:
        """Sample checks for given circuit.

        :param circ: The circuit for which checks should be sampled.
        :return: Pauli checks sampled
        """
        pass

    def add_pauli_checks_to_circbox(
        self,
        circuit: Circuit,
    ) -> Tuple[Circuit, set[Bit]]:
        """Add checks to all subcircuits labeled "Clifford Subcircuit".

        :param circuit: Circuit to add checks to.
        :return: Circuit with checks added.
        """

        # Initialise new circuit and add matching qubits.
        pauli_check_circuit = Circuit()
        for qubit in circuit.qubits:
            pauli_check_circuit.add_qubit(qubit)
        for bit in circuit.bits:
            pauli_check_circuit.add_bit(bit)

        ancilla_count = 0

        postselect_bits = set()

        # Add each command in the circuit, wrapped by checks
        # if the command is a circbox named 'Clifford Subcircuit'
        for command in circuit.get_commands():
            # Add barriers and check if appropriate
            if (
                (command.op.type == OpType.CircBox)
                and (cast(CircBox, command.op).get_circuit().name is not None)
                and (
                    str(cast(CircBox, command.op).get_circuit().name).startswith(
                        "Clifford Subcircuit"
                    )
                )
            ):
                clifford_subcircuit = self._decompose_clifford_subcircuit_box(command)

                # List of Paulis to be used as checks before the subcircuit.
                start_stabiliser_list = self.sample(
                    circ=clifford_subcircuit,
                )

                # TODO: check that register names do not already exist
                control_qubit_list = [
                    Qubit(name="ancilla", index=i)
                    for i in range(
                        ancilla_count, ancilla_count + len(start_stabiliser_list)
                    )
                ]
                ancilla_count += len(start_stabiliser_list)

                # Add ancilla qubits and check gates
                for start_stabiliser, control_qubit in zip(
                    start_stabiliser_list, control_qubit_list
                ):
                    pauli_check_circuit.add_qubit(control_qubit)

                    pauli_check_circuit.add_barrier(command.args + [control_qubit])
                    pauli_check_circuit.H(
                        control_qubit,
                        opgroup="ancilla superposition",
                    )

                    stabiliser_circuit = start_stabiliser.get_control_circuit(
                        control_qubit=control_qubit
                    )
                    pauli_check_circuit.append(
                        circuit=stabiliser_circuit,
                    )

                    pauli_check_circuit.add_barrier(command.args + [control_qubit])

                end_stabiliser_list = [
                    start_stabiliser.dagger
                    for start_stabiliser in start_stabiliser_list
                ]

            # Add original command
            pauli_check_circuit.add_gate(command.op, command.args)

            # Add barriers and checks if appropriate.
            if (command.op.type == OpType.CircBox) and (
                cast(CircBox, command.op).get_circuit().name == "Clifford Subcircuit"
            ):
                # For each check, commute the pauli through
                # the clifford circuit.
                for end_stabiliser, control_qubit in zip(
                    reversed(end_stabiliser_list), reversed(control_qubit_list)
                ):
                    for clifford_command in clifford_subcircuit.get_commands():
                        if clifford_command.op.type == OpType.Barrier:
                            continue

                        # TODO: an error would be raised here if clifford_command
                        # is not Clifford. It could be worth raising a clearer
                        # error.
                        end_stabiliser.apply_gate(
                            op=clifford_command.op,
                            qubits=clifford_command.qubits,
                        )

                    pauli_check_circuit.add_barrier(command.args + [control_qubit])

                    # Add the end check.
                    stabiliser_circuit = end_stabiliser.get_control_circuit(
                        control_qubit=control_qubit
                    )
                    pauli_check_circuit.append(
                        circuit=stabiliser_circuit,
                    )
                    pauli_check_circuit.H(
                        control_qubit,
                        opgroup="ancilla superposition",
                    )
                    pauli_check_circuit.add_barrier(command.args + [control_qubit])

                    measure_bit = Bit(
                        name="ancilla_measure",
                        index=control_qubit.index,
                    )

                    postselect_bits.add(measure_bit)

                    pauli_check_circuit.add_bit(id=measure_bit)
                    pauli_check_circuit.Measure(
                        qubit=control_qubit,
                        bit=measure_bit,
                    )

        return pauli_check_circuit, postselect_bits

    @staticmethod
    def _decompose_clifford_subcircuit_box(clifford_subcircuit_box: Command) -> Circuit:
        """Decompose command by extracting circuit and relabelling qubits to
        match those of the command.

        :param clifford_subcircuit_box: Command to decompose.
        :return: Decomposed command.
        """
        clifford_subcircuit = cast(CircBox, clifford_subcircuit_box.op).get_circuit()
        qubit_map = {
            cast(UnitID, q_subcirc): q_orig
            for q_subcirc, q_orig in zip(
                clifford_subcircuit.qubits, clifford_subcircuit_box.args
            )
        }
        clifford_subcircuit.rename_units(qubit_map)

        return clifford_subcircuit


class DeterministicZPauliSampler(PauliSampler):
    """Deterministic sampler, always returning Z Pauli string."""

    def sample(self, circ: Circuit) -> List[QermitPauli]:
        """Return Z Pauli string of length equal to the circuit.

        :param circ: Circuit to sample Pauli for.
        :return: Z Pauli string of length equal to the circuit.
        """
        return [
            QermitPauli(
                QubitPauliTensor(
                    string=QubitPauliString(
                        map={qubit: Pauli.Z for qubit in circ.qubits}
                    ),
                    coeff=1,
                )
            )
        ]


class DeterministicXPauliSampler(PauliSampler):
    """Deterministic sampler, always returning X Pauli string."""

    def sample(self, circ: Circuit) -> List[QermitPauli]:
        """Return X Pauli string of length equal to the circuit.

        :param circ: Circuit to sample Pauli for.
        :return: X Pauli string of length equal to the circuit.
        """
        return [
            QermitPauli(
                QubitPauliTensor(
                    string=QubitPauliString(
                        map={qubit: Pauli.X for qubit in circ.qubits}
                    ),
                    coeff=1,
                )
            )
        ]


class RandomPauliSampler(PauliSampler):
    """Sampler returning random Pauli of appropriate length."""

    def __init__(
        self, n_checks: int, rng: Generator = numpy.random.default_rng()
    ) -> None:
        """
        :param n_checks: The number of checks to sample
        :param rng: Randomness generator,
            defaults to numpy.random.default_rng()
        """
        self.rng = rng
        self.n_checks = n_checks

    def sample(
        self,
        circ: Circuit,
    ) -> List[QermitPauli]:
        """Sample random Pauli of length equal to the size of the circuit.

        :param circ: Circuit to sample Pauli check for.
        :return: Random Pauli of length equal to the size of the circuit.
        """
        # TODO: Make sure sampling is done without replacement
        stabiliser_list: List[QermitPauli] = []
        while len(stabiliser_list) < self.n_checks:
            qpt = QubitPauliTensor(
                string=QubitPauliString(
                    map={
                        qubit: self.rng.choice(
                            numpy.array([Pauli.X, Pauli.Y, Pauli.Z, Pauli.I])
                        )
                        for qubit in circ.qubits
                    }
                ),
                coeff=1,
            )
            if qpt != QubitPauliTensor():
                stabiliser_list.append(QermitPauli(qpt))

        return stabiliser_list


class OptimalPauliSampler(PauliSampler):
    """
    Samples pauli check based on a noise model. Simulates the noise models
    action on clifford subcircuits in order to select checks.
    """

    def __init__(self, noise_model: NoiseModel, n_checks: int) -> None:
        """
        :param noise_model: The noise model to optimally pick pauli
            checks for.
        :param n_checks: The number of checks to sample.
        """
        self.noise_model = noise_model
        self.n_checks = n_checks

    def sample(
        self,
        circ: Circuit,
    ) -> List[QermitPauli]:
        """Samples checks for the given circuit.

        :param circ: The circuit to sample checks for.
        :return: Optimal Pauli checks.
        """
        # TODO: assert that the registers match in this case

        error_counter = self.noise_model.get_effective_pre_error_distribution(
            cliff_circ=circ,
        )

        # print("effective error distribution", error_counter.distribution)

        total_commute_prob = 0.0
        total_n_pauli = 0

        # smallest_commute_prob stores the proportion of shots which will
        # still have errors.
        smallest_commute_prob = 1.0
        # TODO: There is probably a better way to search through this space.
        # Here are some ideas:
        #   -   It's better to prioritise checks that require fewer gates.
        #       Here I am checking I last as it requires the fewest gates.
        #       However note that IYY is checked after YII so IYY may be
        #       picked even though YII is lighter in the case that they
        #       have the same probability.
        #   -   It may be redundant to check Pauli.I? If a string is selected
        #       with equal probability it may be worth it though.
        #   -   Eventually we may have to select Pauli strings at random,
        #       but i'm not sure at what size that will be necessary.
        for pauli_string_list in combinations(
            product([Pauli.Y, Pauli.X, Pauli.Z, Pauli.I], repeat=circ.n_qubits),
            self.n_checks,
        ):
            if tuple([Pauli.I] * circ.n_qubits) in pauli_string_list:
                continue

            qubit_pauli_string_list = [
                QubitPauliString(qubits=circ.qubits, paulis=pauli_string)
                for pauli_string in pauli_string_list
            ]
            commute_prob = 0.0
            for error, prob in error_counter.distribution.items():
                if all(
                    error.commutes_with(qubit_pauli_string)
                    for qubit_pauli_string in qubit_pauli_string_list
                ):
                    commute_prob += prob
            # print(pauli_string_list, commute_prob)
            if smallest_commute_prob >= commute_prob:
                smallest_commute_prob = commute_prob
                smallest_commute_prob_pauli_list = qubit_pauli_string_list

            total_commute_prob += commute_prob
            total_n_pauli += 1

        # average_commute_prob = total_commute_prob / total_n_pauli
        # print("smallest_commute_prob_pauli_list", smallest_commute_prob_pauli_list)
        # print("smallest_commute_prob", smallest_commute_prob)
        # print("average commute_prob", average_commute_prob)

        # if (average_commute_prob == 0) or (
        #     abs(1 - (smallest_commute_prob / average_commute_prob)) < 0.1
        # ):
        #     warnings.warn(
        #         "The smallest commute probability is close to the average. "
        #         + "Random check sampling will probably work just as well."
        #     )

        return [
            QermitPauli(QubitPauliTensor(string=smallest_commute_prob_pauli, coeff=1))
            for smallest_commute_prob_pauli in smallest_commute_prob_pauli_list
        ]

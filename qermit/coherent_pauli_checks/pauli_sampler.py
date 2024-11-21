import warnings
from abc import ABC, abstractmethod
from itertools import combinations, product
from typing import List, cast

import numpy.random
from numpy.random import Generator
from pytket import Circuit
from pytket.circuit import Bit, CircBox, Command, OpType, Qubit
from pytket.pauli import Pauli, QubitPauliString

from qermit.noise_model.noise_model import NoiseModel
from qermit.noise_model.qermit_pauli import QermitPauli


class PauliSampler(ABC):
    @abstractmethod
    def sample(self, circ: Circuit) -> List[QermitPauli]:  # pragma: no cover
        pass


class DeterministicZPauliSampler(PauliSampler):
    def sample(self, circ: Circuit) -> List[QermitPauli]:
        return [
            QermitPauli(
                Z_list=[1] * circ.n_qubits,
                X_list=[0] * circ.n_qubits,
                qubit_list=circ.qubits,
            )
        ]


class DeterministicXPauliSampler(PauliSampler):
    def sample(self, circ: Circuit) -> List[QermitPauli]:
        return [
            QermitPauli(
                Z_list=[0] * circ.n_qubits,
                X_list=[1] * circ.n_qubits,
                qubit_list=circ.qubits,
            )
        ]


class RandomPauliSampler(PauliSampler):
    def __init__(
        self, n_checks: int, rng: Generator = numpy.random.default_rng()
    ) -> None:
        self.rng = rng
        self.n_checks = n_checks

    def sample(
        self,
        circ: Circuit,
    ) -> List[QermitPauli]:
        # TODO: Make sure sampling is done without replacement

        stabiliser_list: List[QermitPauli] = []
        while len(stabiliser_list) < self.n_checks:
            Z_list = [self.rng.integers(2) for _ in circ.qubits]
            X_list = [self.rng.integers(2) for _ in circ.qubits]

            # Avoids using the identity string as it commutes with all errors
            if any(Z == 1 for Z in Z_list) or any(X == 1 for X in X_list):
                stabiliser_list.append(
                    QermitPauli(
                        Z_list=Z_list,
                        X_list=X_list,
                        qubit_list=circ.qubits,
                    )
                )

        return stabiliser_list


class OptimalPauliSampler(PauliSampler):
    def __init__(self, noise_model: NoiseModel, n_checks: int) -> None:
        self.noise_model = noise_model
        self.n_checks = n_checks

    def sample(
        self,
        circ: Circuit,
    ) -> List[QermitPauli]:
        # TODO: assert that the registers match in this case

        error_counter = self.noise_model.get_effective_pre_error_distribution(circ)

        print("effective error distribution", error_counter.distribution)

        total_commute_prob = 0.0
        total_n_pauli = 0

        # smallest_commute_prob stores the proportion of shots which will
        # still have errors.
        smallest_commute_prob = 1.0
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

        average_commute_prob = total_commute_prob / total_n_pauli
        print("smallest_commute_prob_pauli_list", smallest_commute_prob_pauli_list)
        print("smallest_commute_prob", smallest_commute_prob)
        print("average commute_prob", average_commute_prob)

        if (average_commute_prob == 0) or (
            abs(1 - (smallest_commute_prob / average_commute_prob)) < 0.1
        ):
            warnings.warn(
                "The smallest commute probability is close to the average. "
                + "Random check sampling will probably work just as well."
            )

        return [
            QermitPauli.from_qubit_pauli_string(smallest_commute_prob_pauli)
            for smallest_commute_prob_pauli in smallest_commute_prob_pauli_list
        ]


class CircuitPauliChecker:
    def __init__(self, pauli_sampler: PauliSampler) -> None:
        self.pauli_sampler = pauli_sampler

    def add_pauli_checks_to_circbox(
        self,
        circuit: Circuit,
    ) -> Circuit:
        pauli_check_circuit = Circuit()
        for qubit in circuit.qubits:
            pauli_check_circuit.add_qubit(qubit)
        for bit in circuit.bits:
            pauli_check_circuit.add_bit(bit)

        ancilla_count = 0

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
                clifford_subcircuit = self.decompose_clifford_subcircuit_box(command)

                start_stabiliser_list = self.pauli_sampler.sample(
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
                    start_stabiliser.dagger()
                    for start_stabiliser in start_stabiliser_list
                ]

            # Add command
            pauli_check_circuit.add_gate(command.op, command.args)

            # Add barriers and checks if appropriate.
            if (command.op.type == OpType.CircBox) and (
                cast(CircBox, command.op).get_circuit().name == "Clifford Subcircuit"
            ):
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
                            clifford_command.op.type,
                            clifford_command.qubits,
                            params=clifford_command.op.params,
                        )

                    pauli_check_circuit.add_barrier(command.args + [control_qubit])

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
                    pauli_check_circuit.add_bit(id=measure_bit)
                    pauli_check_circuit.Measure(
                        qubit=control_qubit,
                        bit=measure_bit,
                    )

        return pauli_check_circuit

    @staticmethod
    def decompose_clifford_subcircuit_box(clifford_subcircuit_box: Command) -> Circuit:
        clifford_subcircuit = cast(CircBox, clifford_subcircuit_box.op).get_circuit()
        qubit_map = {
            q_subcirc: q_orig
            for q_subcirc, q_orig in zip(
                clifford_subcircuit.qubits, clifford_subcircuit_box.args
            )
        }
        clifford_subcircuit.rename_units(qubit_map)

        return clifford_subcircuit

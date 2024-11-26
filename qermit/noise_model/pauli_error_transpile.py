from typing import cast

from pytket import Circuit, OpType, Qubit
from pytket.passes import BasePass, CustomPass
from pytket.pauli import Pauli

from .noise_model import NoiseModel


def PauliErrorTranspile(noise_model: NoiseModel) -> BasePass:
    """Generates compiler pass which adds coherent noise to a circuit.

    :param noise_model: Model describing the noise to be added. Should be
        a Pauli noise model.
    :return: Compiler pass adding random coherent Pauli noise.
    """

    def add_gates(circuit: Circuit) -> Circuit:
        """Function adding random coherent Pauli errors to a circuit.

        :param circuit: Circuit to which errors are added.
        :raises Exception: Raised if the noise model is not a Pauli one.
        :return: Circuit with additional noise operations.
        """

        # Initialise circuit with the same registers as input.
        noisy_circuit = Circuit()
        for q_register in circuit.q_registers:
            noisy_circuit.add_q_register(q_register)
        for c_register in circuit.c_registers:
            noisy_circuit.add_c_register(c_register)

        # Add each command in the original circuit,
        # and a pauli error if appropriate.
        for command in circuit.get_commands():
            if command.op.type == OpType.Barrier:
                noisy_circuit.add_barrier(command.args)
            else:
                noisy_circuit.add_gate(command.op, command.args)

            # If command has noise model defined, add a random error
            if command.op.type in noise_model.noisy_gates:
                # Sample a random error, which may be None
                error = noise_model.get_error_distribution(command.op.type).sample()
                if error is not None:
                    for qubit, pauli in zip(command.args, error):
                        if pauli in [Pauli.X, OpType.X]:
                            noisy_circuit.X(cast(Qubit, qubit), opgroup="noisy")
                        elif pauli in [Pauli.Z, OpType.Z]:
                            noisy_circuit.Z(cast(Qubit, qubit), opgroup="noisy")
                        elif pauli in [Pauli.Y, OpType.Y]:
                            noisy_circuit.Y(cast(Qubit, qubit), opgroup="noisy")
                        elif pauli in [Pauli.I]:
                            pass
                        else:
                            raise Exception(
                                "Not a Pauli noise model." + f" Contains {pauli} error"
                            )

        return noisy_circuit

    return CustomPass(add_gates)

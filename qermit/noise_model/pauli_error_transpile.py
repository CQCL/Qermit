from pytket.pauli import Pauli  # type: ignore
from pytket.passes import CustomPass  # type: ignore
from pytket import Circuit, OpType


def PauliErrorTranspile(noise_model):

    def add_gates(circuit):

        noisy_circuit = Circuit()
        for register in circuit.q_registers:
            noisy_circuit.add_q_register(register)
        for register in circuit.c_registers:
            noisy_circuit.add_c_register(register)

        for command in circuit.get_commands():

            if command.op.type == OpType.Barrier:
                noisy_circuit.add_barrier(command.args)
            else:
                noisy_circuit.add_gate(command.op, command.args)

            if command.op.type in noise_model.noisy_gates:
                error = noise_model.get_error_distribution(
                    command.op.type
                ).sample()
                if error is not None:
                    for qubit, pauli in zip(
                        command.args,
                        error
                    ):
                        if pauli in [Pauli.X, OpType.X]:
                            noisy_circuit.X(qubit, opgroup='noisy')
                        elif pauli in [Pauli.Z, OpType.Z]:
                            noisy_circuit.Z(qubit, opgroup='noisy')
                        elif pauli in [Pauli.Y, OpType.Y]:
                            noisy_circuit.Y(qubit, opgroup='noisy')
                        elif pauli in [Pauli.I]:
                            pass
                        else:
                            raise Exception(
                                "Not a Pauli noise model."
                                + f" Contains {pauli} error"
                            )

        return noisy_circuit

    return CustomPass(add_gates)

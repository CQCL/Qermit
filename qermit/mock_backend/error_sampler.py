from pytket import OpType
from .stabiliser import Stabiliser
from collections import Counter

# TODO: there should of course be checks all over the place to see
# if the circuits used are Clifford
class ErrorSampler:

    def __init__(self, noise_model):

        self.noise_model = noise_model

    def counter_propagate(self, cliff_circ, n_counts=1000, **kwargs):

        error_counter = Counter()

        # There is some time wasted here, if for example there is no error in
        # back_propagate_random_error. There may be a saving to be made here
        # if there errors are sampled before the back propagation occurs?
        for _ in range(n_counts):
            stabiliser = self.random_propagate(cliff_circ, **kwargs)

            if not stabiliser.is_identity():
                error_counter.update([stabiliser])

        # print(error_counter.total())

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
            raise Exception(f"direction must be 'backward' or 'forward'. Is {direction}")

        for command in command_list:

            if command.op.type in [OpType.Measure, OpType.Barrier]:
                continue

            if direction == 'forward':

                stabiliser.apply_gate(
                    op_type=command.op.type,
                    qubits=command.args,
                    params=command.op.params,
                )

            if command.op.type in self.noise_model.noisy_gates:

                error_distribution = self.noise_model.get_error_distribution(
                    optype=command.op.type
                )
                error = error_distribution.sample()

                if error != None:
                    for pauli, qubit in zip(error, command.args):
                        if direction == 'backward':
                            stabiliser.pre_apply_pauli(pauli=pauli, qubit=qubit)
                        elif direction == 'forward':
                            stabiliser.post_apply_pauli(pauli=pauli, qubit=qubit)
                        else:
                            raise Exception("How did you get here?")
                        

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
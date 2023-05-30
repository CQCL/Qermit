from pytket.pauli import QubitPauliString, Pauli  # type: ignore
from pytket.circuit import Qubit, OpType, Circuit  # type: ignore


class Stabiliser:
    """For the manipulation of stabilisers. In particular, how they are
    changed by the action of Clifford circuits.
    """

    phase_dict = {
        0: 1 + 0j,
        1: 0 + 1j,
        2: -1 + 0j,
        3: 0 - 1j,
    }

    def __init__(self, Z_list: list[int], qubit_list: list[Qubit]):
        """Initialisation is by a list of qubits, and a list of 0, 1
        values indicating that a Z operator acts there.

        :param Z_list: 0 indicates no Z, 1 indicates Z.
        :type Z_list: list[int]
        :param qubit_list: List of qubits on which the stabiliser acts.
        :type qubit_list: list[Qubit]
        """

        assert all([Z in {0, 1} for Z in Z_list])
        assert len(Z_list) == len(qubit_list)

        self.Z_list = {qubit: Z for qubit, Z in zip(qubit_list, Z_list)}
        # The equivalent X list is initialised in 0. As such only Z
        # stabilisers can be initialised. This could be generalised if
        # necessary.
        self.X_list = {qubit: 0 for qubit in qubit_list}
        self.phase = 0
        self.qubit_list = qubit_list

    def apply_circuit(self, circuit: Circuit):
        """Apply a circuit to a stabiliser. The circuit should be
        a Clifford circuit.

        :param circuit: Circuit to be applied.
        :type circuit: Circuit
        """

        for command in circuit.get_commands():
            self.apply_gate(op_type=command.op.type, qubits=command.qubits)

    def apply_gate(self, op_type: OpType, qubits: list[Qubit]):
        """Apply operation of given type to given qubit in stabiliser. At
        present the recognised operation types are H, S, CX, Z, Sdg,
        X, Y, CZ, SWAP, and Barrier.

        :param op_type: Type of operator to be applied.
        :type op_type: OpType
        :param qubits: Qubits to which operator is applied.
        :type qubits: list[Qubit]
        :raises Exception: Raised if operator is not recognised.
        """

        if op_type == OpType.H:
            self.H(qubit=qubits[0])
        elif op_type == OpType.S:
            self.S(qubit=qubits[0])
        elif op_type == OpType.CX:
            self.CX(control_qubit=qubits[0], target_qubit=qubits[1])
        elif op_type == OpType.Z:
            self.S(qubit=qubits[0])
            self.S(qubit=qubits[0])
        elif op_type == OpType.Sdg:
            self.S(qubit=qubits[0])
            self.S(qubit=qubits[0])
            self.S(qubit=qubits[0])
        elif op_type == OpType.X:
            self.H(qubit=qubits[0])
            self.apply_gate(op_type=OpType.Z, qubits=qubits)
            self.H(qubit=qubits[0])
        elif op_type == OpType.Y:
            self.apply_gate(op_type=OpType.Z, qubits=qubits)
            self.apply_gate(op_type=OpType.X, qubits=qubits)
        elif op_type == OpType.CZ:
            self.H(qubit=qubits[1])
            self.CX(control_qubit=qubits[0], target_qubit=qubits[1])
            self.H(qubit=qubits[1])
        elif op_type == OpType.SWAP:
            self.CX(control_qubit=qubits[0], target_qubit=qubits[1])
            self.CX(control_qubit=qubits[1], target_qubit=qubits[0])
            self.CX(control_qubit=qubits[0], target_qubit=qubits[1])
        elif op_type == OpType.Barrier:
            pass
        else:
            raise Exception(
                f"{op_type} is an unrecognised gate type."
                + "Please use only Clifford gates."
            )

    def __str__(self) -> str:
        stab_str = f"X | {self.X_list}"
        stab_str += f"\nZ | {self.Z_list}"
        stab_str += f"\nphase = {self.phase_dict[self.phase]}"
        return stab_str

    def S(self, qubit: Qubit):
        """Act S operation.

        :param qubit: Qubit in stabiliser onto which S is acted.
        :type qubit: Qubit
        """

        self.Z_list[qubit] += self.X_list[qubit]
        self.Z_list[qubit] %= 2
        self.phase += self.X_list[qubit]
        self.phase %= 4

    def H(self, qubit: Qubit):
        """Act H operation.

        :param qubit: Qubit in stabiliser on which H is acted.
        :type qubit: Qubit
        """

        self.phase += 2 * self.X_list[qubit] * self.Z_list[qubit]
        self.phase %= 4

        temp_X = self.X_list[qubit]
        self.X_list[qubit] = self.Z_list[qubit]
        self.Z_list[qubit] = temp_X

    def CX(self, control_qubit: Qubit, target_qubit: Qubit):
        """Act CX operation.

        :param control_qubit: Control qubit of CX gate.
        :type control_qubit: Qubit
        :param target_qubit: Target qubit of CX gate.
        :type target_qubit: Qubit
        """

        self.Z_list[control_qubit] += self.Z_list[target_qubit]
        self.Z_list[control_qubit] %= 2
        self.X_list[target_qubit] += self.X_list[control_qubit]
        self.X_list[target_qubit] %= 2

    def get_control_circuit(self, control_qubit: Qubit) -> Circuit:
        """Circuit which acts stabiliser.

        :return: Circuit acting stabiliser.
        :rtype: Circuit
        """

        circ = Circuit()
        circ.add_qubit(control_qubit)
        for qubit in self.qubit_list:
            circ.add_qubit(id=qubit)
            if self.Z_list[qubit] == 1:
                circ.CZ(
                    control_qubit=control_qubit,
                    target_qubit=qubit,
                )
            if self.X_list[qubit] == 1:
                circ.CX(
                    control_qubit=control_qubit,
                    target_qubit=qubit,
                )
        circ.add_phase(a=self.phase / 2)

        return circ

    @property
    def circuit(self) -> Circuit:
        """Circuit which acts stabiliser.

        :return: Circuit acting stabiliser.
        :rtype: Circuit
        """

        circ = Circuit()
        for qubit in self.qubit_list:
            circ.add_qubit(id=qubit)
            if self.Z_list[qubit] == 1:
                circ.Z(qubit)
            if self.X_list[qubit] == 1:
                circ.X(qubit)
        circ.add_phase(a=self.phase / 2)

        return circ

    @property
    def qubit_pauli_string(self) -> tuple[QubitPauliString, complex]:
        """Qubit pauli string corresponding to stabiliser,
        along with the appropriate phase.

        :return: Pauli string and phase corresponding to stabiliser.
        :rtype: tuple[QubitPauliString, complex]
        """

        operator_phase = self.phase
        paulis = []
        for X, Z in zip(self.X_list.values(), self.Z_list.values()):
            if X == 0 and Z == 0:
                paulis.append(Pauli.I)
            elif X == 1 and Z == 0:
                paulis.append(Pauli.X)
            elif X == 0 and Z == 1:
                paulis.append(Pauli.Z)
            elif X == 1 and Z == 1:
                paulis.append(Pauli.Y)
                operator_phase += 3
                operator_phase %= 4

        qubit_pauli_string = QubitPauliString(
            qubits=self.qubit_list, paulis=paulis
        )

        return qubit_pauli_string, self.phase_dict[operator_phase]


class PauliSampler:

    def sample(self, **kwargs):
        pass


class DeterministicPauliSampler(PauliSampler):

    def __init__(self):
        pass

    def sample(self, qubit_list):
        return Stabiliser(
            Z_list=[1] * len(qubit_list),
            qubit_list=qubit_list,
        )

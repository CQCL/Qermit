from pytket.pauli import QubitPauliString, Pauli  # type: ignore
from pytket.circuit import Qubit, OpType, Circuit  # type: ignore
import math
import numpy as np
from numpy.random import Generator
from typing import List
from pytket import Qubit


class Stabiliser:
    """For the manipulation of stabilisers. In particular, how they are
    changed by the action of Clifford circuits. Note that each term in the
    tensor product of the stabiliser should be thought of as:
    (i)^{phase}X^{X_list}Z^{Z_list}
    """

    phase_dict = {
        0: 1+0j,
        1: 0+1j,
        2: -1+0j,
        3: 0-1j,
    }

    def __init__(
        self,
        Z_list: list[int],
        X_list: list[int],
        qubit_list: list[Qubit],
        phase=0
    ):
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
        self.X_list = {qubit: X for qubit, X in zip(qubit_list, X_list)}
        self.phase = phase
        self.qubit_list = qubit_list

    def is_measureable(self, qubit_list:List[Qubit]):
        if not all(qubit in self.qubit_list for qubit in qubit_list):
            raise Exception(f"{qubit_list} is not a subset of {self.qubit_list}.")
        return any(self.X_list[qubit] == 1 for qubit in qubit_list)

    # def contains(self, sub_stabiliser):
    #     return (
    #         all(
    #             qubit in self.qubit_list
    #             for qubit in sub_stabiliser.qubit_list
    #         ) and all(
    #             sub_stabiliser.Z_list[qubit] == self.Z_list[qubit]
    #             for qubit in sub_stabiliser.qubit_list
    #         ) and all(
    #             sub_stabiliser.X_list[qubit] == self.X_list[qubit]
    #             for qubit in sub_stabiliser.qubit_list
    #         )
    #     )

    def is_identity(self):
        return all(
            Z == 0 for Z in self.Z_list.values()
        ) and all(
            X == 0 for X in self.X_list.values()
        )

    @classmethod
    def random_stabiliser(
        cls,
        qubit_list: list[Qubit],
        rng: Generator = np.random.default_rng(),
    ):

        return cls(
            Z_list=list(rng.integers(2, size=len(qubit_list))),
            X_list=list(rng.integers(2, size=len(qubit_list))),
            qubit_list=qubit_list,
        )

    def dagger(self):

        # the phase is the conjugate of the original
        phase = self.phase
        phase += 2*(self.phase % 2)

        Z_list = list(self.Z_list.values())
        X_list = list(self.X_list.values())
        for Z, X in zip(Z_list, X_list):
            phase += 2*Z*X
        phase %= 4

        return Stabiliser(
            Z_list=Z_list,
            X_list=X_list,
            qubit_list=self.qubit_list,
            phase=phase,
        )

    @classmethod
    def from_qubit_pauli_string(cls, qps):

        Z_list = []
        X_list = []
        phase = 0
        qubit_list = []

        for pauli in qps.to_list():

            qubit = Qubit(name=pauli[0][0], index=pauli[0][1])
            qubit_list.append(qubit)

            if pauli[1] in ['Z', 'Y']:
                Z_list.append(1)
            else:
                Z_list.append(0)

            if pauli[1] in ['X', 'Y']:
                X_list.append(1)
            else:
                X_list.append(0)

            if pauli[1] == 'Y':
                phase += 1
                phase %= 4

        return cls(
            Z_list=Z_list,
            X_list=X_list,
            qubit_list=qubit_list,
            phase=phase,
        )

    def __hash__(self):
        key = (
            *list(self.Z_list.values()),
            *list(self.X_list.values()),
            *self.qubit_list,
            self.phase,
        )
        return hash(key)

    def __str__(self) -> str:
        # stab_str = f"X | {self.X_list}"
        # stab_str += f"\nZ | {self.Z_list}"
        # stab_str += f"\nphase = {self.phase_dict[self.phase]}"
        # return stab_str

        qubit_pauli_string, operator_phase = self.qubit_pauli_string
        return f"{qubit_pauli_string}, {operator_phase}"

    def __eq__(self, other):

        if (
            sorted(list(self.X_list.keys()))
            != sorted(list(other.X_list.keys()))
        ):
            return False
        if not all(
            self.X_list[quibt] == other.X_list[quibt]
            for quibt in self.X_list.keys()
        ):
            return False
        if (
            sorted(list(self.Z_list.keys()))
            != sorted(list(other.Z_list.keys()))
        ):
            return False
        if not all(
            self.Z_list[quibt] == other.Z_list[quibt]
            for quibt in self.X_list.keys()
        ):
            return False
        if self.phase != other.phase:
            return False

        return True

    def apply_circuit(self, circuit: Circuit):
        """Apply a circuit to a stabiliser. The circuit should be
        a Clifford circuit.

        :param circuit: Circuit to be applied.
        :type circuit: Circuit
        """

        for command in circuit.get_commands():
            if command.op.type == OpType.Barrier:
                continue
            self.apply_gate(
                op_type=command.op.type,
                qubits=command.qubits,
                params=command.op.params,
            )

    def apply_gate(self, op_type: OpType, qubits: list[Qubit], **kwargs):
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
        elif op_type == OpType.PhasedX:
            params = kwargs.get("params", None)
            if all(
                math.isclose(param % 0.5, 0) or math.isclose(param % 0.5, 0.5)
                for param in params
            ):
                self.apply_gate(OpType.Rz, qubits=qubits, params=[-params[1]])
                self.apply_gate(OpType.Rx, qubits=qubits, params=[params[0]])
                self.apply_gate(OpType.Rz, qubits=qubits, params=[params[1]])
            else:
                raise Exception(
                    f"{params} are not clifford angles for " +
                    "PhasedX."
                )
        elif op_type == OpType.Rz:
            params = kwargs.get("params", None)
            angle = params[0]
            if math.isclose(angle % 0.5, 0) or math.isclose(angle % 0.5, 0.5):
                angle = round(angle, 1)
                for _ in range(int((angle % 2)//0.5)):
                    self.S(qubit=qubits[0])
            else:
                raise Exception(
                    f"{angle} is not a clifford angle."
                )
        elif op_type == OpType.Rx:
            params = kwargs.get("params", None)
            angle = params[0]
            if math.isclose(angle % 0.5, 0) or math.isclose(angle % 0.5, 0.5):
                angle = round(angle, 1)
                self.H(qubit=qubits[0])
                for _ in range(int((angle % 2)//0.5)):
                    self.S(qubit=qubits[0])
                self.H(qubit=qubits[0])
            else:
                raise Exception(
                    f"{angle} is not a clifford angle."
                )
        elif op_type == OpType.ZZMax:
            self.CX(control_qubit=qubits[0], target_qubit=qubits[1])
            self.S(qubit=qubits[1])
            self.CX(control_qubit=qubits[0], target_qubit=qubits[1])
        elif op_type == OpType.ZZPhase:
            params = kwargs.get("params", None)
            angle = params[0]
            if math.isclose(angle % 0.5, 0) or math.isclose(angle % 0.5, 0.5):
                angle = round(angle, 1)
                for _ in range(int((angle % 2)//0.5)):
                    self.apply_gate(op_type=OpType.ZZMax, qubits=qubits)
            else:
                raise Exception(
                    f"{angle} is not a clifford angle."
                )
        elif op_type == OpType.Barrier:
            pass
        else:
            raise Exception(
                f"{op_type} is an unrecognised gate type. " +
                "Please use only Clifford gates."
            )

    def S(self, qubit: Qubit):
        """Act S operation on stabiliser. In particular this transforms
        the stabiliser (i)^{phase}X^{X_liist}Z^{Z_list} to
        (i)^{phase}SX^{X_liist}Z^{Z_list}S^{dagger}.

        :param qubit: Qubit in stabiliser onto which S is acted.
        :type qubit: Qubit
        """

        self.Z_list[qubit] += self.X_list[qubit]
        self.Z_list[qubit] %= 2
        self.phase += self.X_list[qubit]
        self.phase %= 4

    def H(self, qubit: Qubit):
        """Act H operation. In particular this transforms
        the stabiliser (i)^{phase}X^{X_liist}Z^{Z_list} to
        H(i)^{phase}X^{X_liist}Z^{Z_list}H^{dagger}.

        :param qubit: Qubit in stabiliser on which H is acted.
        :type qubit: Qubit
        """

        self.phase += 2*self.X_list[qubit]*self.Z_list[qubit]
        self.phase %= 4

        temp_X = self.X_list[qubit]
        self.X_list[qubit] = self.Z_list[qubit]
        self.Z_list[qubit] = temp_X

    def CX(self, control_qubit: Qubit, target_qubit: Qubit):
        """Act CX operation. In particular this transforms
        the stabiliser (i)^{phase}X^{X_liist}Z^{Z_list} to
        CX(i)^{phase}X^{X_liist}Z^{Z_list}CX^{dagger}.

        :param control_qubit: Control qubit of CX gate.
        :type control_qubit: Qubit
        :param target_qubit: Target qubit of CX gate.
        :type target_qubit: Qubit
        """

        self.Z_list[control_qubit] += self.Z_list[target_qubit]
        self.Z_list[control_qubit] %= 2
        self.X_list[target_qubit] += self.X_list[control_qubit]
        self.X_list[target_qubit] %= 2

    def pre_multiply(self, stabiliser):

        for qubit in self.qubit_list:
            if stabiliser.X_list[qubit]:
                self.pre_apply_X(qubit)
            if stabiliser.Z_list[qubit]:
                self.pre_apply_Z(qubit)
            self.phase += stabiliser.phase
            self.phase %= 4

    def pre_apply_pauli(self, pauli, qubit):

        if pauli in [Pauli.X, OpType.X]:
            self.pre_apply_X(qubit)
        elif pauli in [Pauli.Z, OpType.Z]:
            self.pre_apply_Z(qubit)
        elif pauli in [Pauli.Y, OpType.Y]:
            self.pre_apply_X(qubit)
            self.pre_apply_Z(qubit)
            self.phase += 1
            self.phase %= 4
        elif pauli == Pauli.I:
            pass
        else:
            raise Exception(
                f"{pauli} is not a Pauli."
            )

    def pre_apply_X(self, qubit: Qubit):

        self.X_list[qubit] += 1
        self.X_list[qubit] %= 2
        self.phase += 2 * self.Z_list[qubit]
        self.phase %= 4

    def pre_apply_Z(self, qubit: Qubit):

        self.Z_list[qubit] += 1
        self.Z_list[qubit] %= 2

    def post_apply_pauli(self, pauli, qubit):

        if pauli in [Pauli.X, OpType.X]:
            self.post_apply_X(qubit)
        elif pauli in [Pauli.Z, OpType.Z]:
            self.post_apply_Z(qubit)
        elif pauli in [Pauli.Y, OpType.Y]:
            self.post_apply_Z(qubit)
            self.post_apply_X(qubit)
            self.phase += 1
            self.phase %= 4
        elif pauli == Pauli.I:
            pass
        else:
            raise Exception(
                f"{pauli} is not a Pauli."
            )

    def post_apply_X(self, qubit: Qubit):

        self.X_list[qubit] += 1
        self.X_list[qubit] %= 2

    def post_apply_Z(self, qubit: Qubit):

        self.Z_list[qubit] += 1
        self.Z_list[qubit] %= 2
        self.phase += 2 * self.X_list[qubit]
        self.phase %= 4

    def get_control_circuit(self, control_qubit: Qubit) -> Circuit:
        """Circuit which acts stabiliser.

        :return: Circuit acting stabiliser.
        :rtype: Circuit
        """

        circ = Circuit()
        circ.add_qubit(control_qubit)
        # TODO: in the case that this is secretly a controlled Y a controlled
        # Y should be applied. Otherwise there is additional noise added in
        # the case of a CY.
        for qubit in self.qubit_list:
            circ.add_qubit(id=qubit)
            if self.Z_list[qubit] == 1:
                circ.CZ(
                    control_qubit=control_qubit,
                    target_qubit=qubit,
                    opgroup='pauli check',
                )
            if self.X_list[qubit] == 1:
                circ.CX(
                    control_qubit=control_qubit,
                    target_qubit=qubit,
                    opgroup='pauli check',
                )

        for _ in range(self.phase):
            circ.S(
                control_qubit,
                opgroup='phase correction',
            )

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
        circ.add_phase(a=self.phase/2)

        return circ

    @property
    def pauli_string(self):

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

        return paulis, self.phase_dict[operator_phase]

    @property
    def qubit_pauli_string(self) -> tuple[QubitPauliString, complex]:
        """Qubit pauli string corresponding to stabiliser,
        along with the appropriate phase.

        :return: Pauli string and phase corresponding to stabiliser.
        :rtype: tuple[QubitPauliString, complex]
        """

        paulis, operator_phase = self.pauli_string

        # operator_phase = self.phase
        # paulis = []
        # for X, Z in zip(self.X_list.values(), self.Z_list.values()):
        #     if X == 0 and Z == 0:
        #         paulis.append(Pauli.I)
        #     elif X == 1 and Z == 0:
        #         paulis.append(Pauli.X)
        #     elif X == 0 and Z == 1:
        #         paulis.append(Pauli.Z)
        #     elif X == 1 and Z == 1:
        #         paulis.append(Pauli.Y)
        #         operator_phase += 3
        #         operator_phase %= 4

        qubit_pauli_string = QubitPauliString(
            qubits=self.qubit_list, paulis=paulis
        )

        # return qubit_pauli_string, self.phase_dict[operator_phase]
        return qubit_pauli_string, operator_phase

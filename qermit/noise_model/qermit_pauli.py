from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Dict, List, Tuple, Union

import numpy as np
from numpy.random import Generator
from pytket.circuit import Circuit, Op, OpType, Qubit
from pytket.pauli import Pauli, QubitPauliString, QubitPauliTensor, pauli_string_mult
from pytket.tableau import UnitaryTableau


class QermitPauli:
    """For the manipulation of Pauli strings. In particular, how they are
    changed by the action of Clifford circuits. Note that each term in the
    tensor product of the Paulis should be thought of as:
    (i)^{phase}X^{X_list}Z^{Z_list}
    """

    phase_dict: Dict[int, complex] = {
        0: 1 + 0j,
        1: 0 + 1j,
        2: -1 + 0j,
        3: 0 - 1j,
    }

    coeff_to_phase = {1 + 0j: 0, 0 + 1j: 1, -1 + 0j: 2, 0 - 1j: 3}

    def __init__(
        self,
        Z_list: List[int],
        X_list: List[int],
        qubit_list: List[Qubit],
        phase: int = 0,
    ):
        """Initialisation is by a list of qubits, and lists of 0, 1
        values indicating that a Z or X operator acts there.

        :param Z_list: 0 indicates no Z, 1 indicates Z.
        :param X_list: 0 indicates no X, 1 indicates X.
        :param qubit_list: List of qubits on which the Pauli acts.
        :param phase: Phase as a power of i
        """

        assert all([Z in {0, 1} for Z in Z_list])
        assert len(Z_list) == len(qubit_list)

        self.Z_list = {qubit: Z for qubit, Z in zip(qubit_list, Z_list)}
        self.X_list = {qubit: X for qubit, X in zip(qubit_list, X_list)}
        self.phase = phase
        self.qubit_list = qubit_list

    def is_measureable(self, qubit_list: List[Qubit]) -> bool:
        """Checks if this Pauli would be measurable on the given qubits in the
        computational bases. That is to say if at least one  Pauli on the given
        qubits anticommutes with Z.

        :param qubit_list: Qubits on which if measurable should be checked.
        :raises Exception: Raised if the given qubits are not contained
            in this Pauli.
        :return: True if at least one Pauli on the given
            qubits anticommutes with Z. False otherwise.
        """
        return any(
            not self.qubit_pauli_tensor.commutes_with(
                QubitPauliTensor(qubit=qubit, pauli=Pauli.Z)
            )
            for qubit in qubit_list
        )

    def reduce_qubits(self, qubit_list: List[Qubit]) -> QermitPauli:
        """Reduces Pauli by removing terms acting on qubits
        in the given list. A new reduced Pauli is created.

        :param qubit_list: Qubits in Pauli which should be removed.
        :return: Reduced Pauli.
        """
        return self.from_qubit_pauli_tensor(
            QubitPauliTensor(
                string=QubitPauliString(
                    map={
                        qubit: pauli
                        for qubit, pauli in self.qubit_pauli_tensor.string.map.items()
                        if qubit not in qubit_list
                    }
                ),
                coeff=self.qubit_pauli_tensor.coeff,
            )
        )

    def dagger(self) -> QermitPauli:
        """Generates the inverse of the Pauli.

        :return: Conjugate transpose of the Pauli.
        """
        return QermitPauli.from_qubit_pauli_tensor(
            qpt=QubitPauliTensor(
                string=self.qubit_pauli_tensor.string,
                coeff=self.qubit_pauli_tensor.coeff.conjugate(),
            )
        )

    @classmethod
    def from_qubit_pauli_tensor(cls, qpt: QubitPauliTensor) -> QermitPauli:
        """Create a Pauli from a qubit pauli string.

        :param qps: Qubit pauli string to be converted to a Pauli.
        :return: Pauli created from qubit pauli string.
        """

        # coeff_to_phase = {1 + 0j: 0, 0 + 1j: 1, -1 + 0j: 2, 0 - 1j: 3}

        Z_list = []
        X_list = []
        phase = cls.coeff_to_phase[qpt.coeff]
        qubit_list = []

        qps = qpt.string

        for pauli in qps.to_list():
            qubit = Qubit(name=pauli[0][0], index=pauli[0][1])
            qubit_list.append(qubit)

            if pauli[1] in ["Z", "Y"]:
                Z_list.append(1)
            else:
                Z_list.append(0)

            if pauli[1] in ["X", "Y"]:
                X_list.append(1)
            else:
                X_list.append(0)

            if pauli[1] == "Y":
                phase += 1
                phase %= 4

        return cls(
            Z_list=Z_list,
            X_list=X_list,
            qubit_list=qubit_list,
            phase=phase,
        )

    def __hash__(self):
        return self.qubit_pauli_tensor.__hash__()

    def __str__(self) -> str:  # pragma: no cover
        return str(self.qubit_pauli_tensor.string) + str(self.qubit_pauli_tensor.coeff)

    def __eq__(self, other: object) -> bool:
        """Checks for equality by checking all qubits match, and that all
        Paulis on those qubits match.

        :param other: Pauli to compare against.
        :return: True is equivalent.
        """

        if not isinstance(other, QermitPauli):
            return False

        return self.qubit_pauli_tensor == other.qubit_pauli_tensor

    def apply_circuit(self, circuit: Circuit):
        """Apply a circuit to a pauli. This is to say commute tha Pauli
        through the circuit. The circuit should be a Clifford circuit.

        :param circuit: Circuit to be applied.
        """

        for command in circuit.get_commands():
            if command.op.type == OpType.Barrier:
                continue

            if command.qubits != command.args:
                raise Exception(
                    "Circuit must be purely quantum."
                    f"The given circuit acts on bits {command.bits}"
                )

            self.apply_gate(
                op=command.op,
                qubits=command.qubits,
            )

    def apply_gate(self, op: Op, qubits: List[Qubit]):
        """Apply operation of given type to given qubit in the pauli. At
        present the recognised operation types are H, S, CX, Z, Sdg,
        X, Y, CZ, SWAP, and Barrier.

        :param op_type: Type of operator to be applied.
        :param qubits: Qubits to which operator is applied.
        :raises Exception: Raised if operator is not recognised.
        """

        if op.type == OpType.H:
            self._H(qubit=qubits[0])
        elif op.type == OpType.S:
            self._S(qubit=qubits[0])
        elif op.type == OpType.CX:
            self._CX(control_qubit=qubits[0], target_qubit=qubits[1])
        elif op.type == OpType.Z:
            self._S(qubit=qubits[0])
            self._S(qubit=qubits[0])
        elif op.type == OpType.Sdg:
            self._S(qubit=qubits[0])
            self._S(qubit=qubits[0])
            self._S(qubit=qubits[0])
        elif op.type == OpType.X:
            self._H(qubit=qubits[0])
            self.apply_gate(op=Op.create(OpType.Z), qubits=qubits)
            self._H(qubit=qubits[0])
        elif op.type == OpType.Y:
            self.apply_gate(op=Op.create(OpType.Z), qubits=qubits)
            self.apply_gate(op=Op.create(OpType.X), qubits=qubits)
        elif op.type == OpType.CZ:
            self._H(qubit=qubits[1])
            self._CX(control_qubit=qubits[0], target_qubit=qubits[1])
            self._H(qubit=qubits[1])
        elif op.type == OpType.SWAP:
            self._CX(control_qubit=qubits[0], target_qubit=qubits[1])
            self._CX(control_qubit=qubits[1], target_qubit=qubits[0])
            self._CX(control_qubit=qubits[0], target_qubit=qubits[1])
        elif op.type == OpType.PhasedX:
            if all(
                math.isclose(param % 0.5, 0) or math.isclose(param % 0.5, 0.5)
                for param in op.params
            ):
                self.apply_gate(op=Op.create(OpType.Rz, [-op.params[1]]), qubits=qubits)
                self.apply_gate(op=Op.create(OpType.Rx, [op.params[0]]), qubits=qubits)
                self.apply_gate(op=Op.create(OpType.Rz, [op.params[1]]), qubits=qubits)
            else:
                raise Exception(
                    f"{op.params} are not clifford angles for " + "PhasedX."
                )
        elif op.type == OpType.Rz:
            angle = op.params[0]
            if math.isclose(angle % 0.5, 0) or math.isclose(angle % 0.5, 0.5):
                angle = round(angle, 1)
                for _ in range(int((angle % 2) // 0.5)):
                    self._S(qubit=qubits[0])
            else:
                raise Exception(f"{angle} is not a clifford angle.")
        elif op.type == OpType.Rx:
            angle = op.params[0]
            if math.isclose(angle % 0.5, 0) or math.isclose(angle % 0.5, 0.5):
                angle = round(angle, 1)
                self._H(qubit=qubits[0])
                for _ in range(int((angle % 2) // 0.5)):
                    self._S(qubit=qubits[0])
                self._H(qubit=qubits[0])
            else:
                raise Exception(f"{angle} is not a clifford angle.")
        elif op.type == OpType.ZZMax:
            self._CX(control_qubit=qubits[0], target_qubit=qubits[1])
            self._S(qubit=qubits[1])
            self._CX(control_qubit=qubits[0], target_qubit=qubits[1])
        elif op.type == OpType.ZZPhase:
            angle = op.params[0]
            if math.isclose(angle % 0.5, 0) or math.isclose(angle % 0.5, 0.5):
                angle = round(angle, 1)
                for _ in range(int((angle % 2) // 0.5)):
                    self.apply_gate(op=Op.create(OpType.ZZMax), qubits=qubits)
            else:
                raise Exception(f"{angle} is not a clifford angle.")
        elif op.type == OpType.Barrier:
            pass
        else:
            raise Exception(
                f"{op.type} is an unrecognised gate type. "
                + "Please use only Clifford gates."
            )

    def _S(self, qubit: Qubit):
        """Act S operation on the pauli. In particular this transforms
        the pauli (i)^{phase}X^{X_liist}Z^{Z_list} to
        (i)^{phase}SX^{X_liist}Z^{Z_list}S^{dagger}.

        :param qubit: Qubit in Pauli onto which S is acted.
        """

        self.Z_list[qubit] += self.X_list[qubit]
        self.Z_list[qubit] %= 2
        self.phase += self.X_list[qubit]
        self.phase %= 4

    def _H(self, qubit: Qubit):
        """Act H operation. In particular this transforms
        the Pauli (i)^{phase}X^{X_liist}Z^{Z_list} to
        H(i)^{phase}X^{X_liist}Z^{Z_list}H^{dagger}.

        :param qubit: Qubit in Pauli on which H is acted.
        """

        self.phase += 2 * self.X_list[qubit] * self.Z_list[qubit]
        self.phase %= 4

        temp_X = self.X_list[qubit]
        self.X_list[qubit] = self.Z_list[qubit]
        self.Z_list[qubit] = temp_X

    def _CX(self, control_qubit: Qubit, target_qubit: Qubit):
        """Act CX operation. In particular this transforms
        the Pauli (i)^{phase}X^{X_liist}Z^{Z_list} to
        CX(i)^{phase}X^{X_liist}Z^{Z_list}CX^{dagger}.

        :param control_qubit: Control qubit of CX gate.
        :param target_qubit: Target qubit of CX gate.
        """

        self.Z_list[control_qubit] += self.Z_list[target_qubit]
        self.Z_list[control_qubit] %= 2
        self.X_list[target_qubit] += self.X_list[control_qubit]
        self.X_list[target_qubit] %= 2

    def pre_apply_pauli(self, pauli: Union[Pauli, OpType], qubit: Qubit):
        """Pre apply by a pauli on a particular qubit.

        :param pauli: Pauli to pre-apply.
        :param qubit: Qubit to apply Pauli to.
        :raises Exception: Raised if pauli is not a pauli operation.
        """

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
            raise Exception(f"{pauli} is not a Pauli.")

    def pre_apply_X(self, qubit: Qubit):
        """Pre-apply X Pauli ito qubit.

        :param qubit: Qubit to which X is pre-applied.
        """

        self.X_list[qubit] += 1
        self.X_list[qubit] %= 2
        self.phase += 2 * self.Z_list[qubit]
        self.phase %= 4

    def pre_apply_Z(self, qubit: Qubit):
        """Pre-apply Z Pauli ito qubit.

        :param qubit: Qubit to which Z is pre-applied.
        """

        self.Z_list[qubit] += 1
        self.Z_list[qubit] %= 2

    def post_apply_pauli(self, pauli: Union[Pauli, OpType], qubit: Qubit):
        """Post apply a Pauli operation.

        :param pauli: Pauli to post-apply.
        :param qubit: Qubit to post-apply pauli to.
        :raises Exception: Raised if pauli is not a Pauli operation.
        """

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
            raise Exception(f"{pauli} is not a Pauli.")

    def post_apply_X(self, qubit: Qubit):
        """Post-apply X Pauli ito qubit.

        :param qubit: Qubit to which X is post-applied.
        """

        self.X_list[qubit] += 1
        self.X_list[qubit] %= 2

    def post_apply_Z(self, qubit: Qubit):
        """Post-apply Z Pauli ito qubit.

        :param qubit: Qubit to which Z is post-applied.
        """

        self.Z_list[qubit] += 1
        self.Z_list[qubit] %= 2
        self.phase += 2 * self.X_list[qubit]
        self.phase %= 4

    def get_control_circuit(self, control_qubit: Qubit) -> Circuit:
        """Controlled circuit which acts Pauli.

        :return: Controlled circuit acting Paulii.
        """

        circ = Circuit()
        circ.add_qubit(control_qubit)
        # TODO: in the case that this is secretly a controlled Y a controlled
        # Y should be applied. Otherwise there is additional noise added in
        # the case of a CY.
        phase = self.coeff_to_phase[self.qubit_pauli_tensor.coeff]
        for qubit, pauli in self.qubit_pauli_tensor.string.map.items():
            circ.add_qubit(id=qubit)

            if pauli == Pauli.Z or pauli == Pauli.Y:
                circ.CZ(
                    control_qubit=control_qubit,
                    target_qubit=qubit,
                    opgroup="pauli check",
                )

            if pauli == Pauli.X or pauli == Pauli.Y:
                circ.CX(
                    control_qubit=control_qubit,
                    target_qubit=qubit,
                    opgroup="pauli check",
                )

            if pauli == Pauli.Y:
                phase += 1
                phase %= 4

        for _ in range(phase):
            circ.S(
                control_qubit,
                opgroup="phase correction",
            )

        return circ

    @property
    def circuit(self) -> Circuit:
        """Circuit which acts Pauli.

        :return: Circuit acting Pauli.
        """

        circ = Circuit()

        phase = self.coeff_to_phase[self.qubit_pauli_tensor.coeff]

        for qubit, pauli in self.qubit_pauli_tensor.string.map.items():
            circ.add_qubit(id=qubit)

            if pauli == Pauli.Z or pauli == Pauli.Y:
                circ.Z(qubit)

            if pauli == Pauli.X or pauli == Pauli.Y:
                circ.X(qubit)

            if pauli == Pauli.Y:
                phase += 1
                phase %= 4

        circ.add_phase(a=phase / 2)

        return circ

    @property
    def qubit_pauli_tensor(self) -> QubitPauliTensor:
        """Return QubitPauliTensor describing Pauli.

        :return: QubitPauliTensor describing Pauli.
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

        return QubitPauliTensor(
            qubits=self.qubit_list,
            paulis=paulis,
            coeff=self.phase_dict[operator_phase],
        )

    @classmethod
    def from_pauli_list(
        cls, pauli_list: List[Pauli], qubit_list: List[Qubit]
    ) -> QermitPauli:
        """Create a QermitPauli from a Pauli iterable.

        :param pauli_iterable: The Pauli iterable to convert.
        :param qubit_list: The qubits on which the resulting pauli will act.
        :return: The pauli corresponding to the given iterable.
        """
        return cls.from_qubit_pauli_tensor(
            qpt=QubitPauliTensor(
                string=QubitPauliString(
                    qubits=qubit_list,
                    paulis=pauli_list,
                ),
                coeff=1,
            ),
        )

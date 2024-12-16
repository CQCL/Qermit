from __future__ import annotations

from typing import Dict, List

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

    def __init__(self, qpt: QubitPauliTensor) -> None:
        self.qubit_list = list(qpt.string.map.keys())

        self.unitary_tableau = UnitaryTableau(nqb=len(self.qubit_list))

        self.qubit_index = {qubit: Qubit(i) for i, qubit in enumerate(self.qubit_list)}
        self.index_qubit = {Qubit(i): qubit for i, qubit in enumerate(self.qubit_list)}

        self.input_pauli_tensor = QubitPauliTensor(
            string=QubitPauliString(
                map={
                    self.qubit_index[qubit]: pauli
                    for qubit, pauli in qpt.string.map.items()
                }
            ),
            coeff=qpt.coeff,
        )

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
        return QermitPauli(
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
        return QermitPauli(
            qpt=QubitPauliTensor(
                string=self.qubit_pauli_tensor.string,
                coeff=self.qubit_pauli_tensor.coeff.conjugate(),
            )
        )

    # @classmethod
    # def from_qubit_pauli_tensor(cls, qpt: QubitPauliTensor) -> QermitPauli:
    #     """Create a Pauli from a qubit pauli string.

    #     :param qps: Qubit pauli string to be converted to a Pauli.
    #     :return: Pauli created from qubit pauli string.
    #     """

    #     Z_list = []
    #     X_list = []
    #     phase = cls.coeff_to_phase[qpt.coeff]
    #     qubit_list = []

    #     qps = qpt.string

    #     for pauli in qps.to_list():
    #         qubit = Qubit(name=pauli[0][0], index=pauli[0][1])
    #         qubit_list.append(qubit)

    #         if pauli[1] in ["Z", "Y"]:
    #             Z_list.append(1)
    #         else:
    #             Z_list.append(0)

    #         if pauli[1] in ["X", "Y"]:
    #             X_list.append(1)
    #         else:
    #             X_list.append(0)

    #         if pauli[1] == "Y":
    #             phase += 1
    #             phase %= 4

    #     return cls(
    #         Z_list=Z_list,
    #         X_list=X_list,
    #         qubit_list=qubit_list,
    #         phase=phase,
    #     )

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

    def apply_gate(self, op: Op, qubits: list[Qubit]):
        if not op.is_clifford():
            raise Exception(f"{op} is not a Clifford operation.")

        if op.is_clifford_type():
            self.unitary_tableau.apply_gate_at_end(
                type=op.type,
                qbs=[self.qubit_index[qubit] for qubit in qubits],
            )

        elif op.type == OpType.Rz:
            for _ in range(int((op.params[0] % 2) // 0.5)):
                self.apply_gate(op=Op.create(OpType.S), qubits=qubits)

        elif op.type == OpType.Rx:
            self.apply_gate(op=Op.create(OpType.H), qubits=qubits)
            for _ in range(int((op.params[0] % 2) // 0.5)):
                self.apply_gate(op=Op.create(OpType.S), qubits=qubits)
            self.apply_gate(op=Op.create(OpType.H), qubits=qubits)

        elif op.type == OpType.PhasedX:
            self.apply_gate(op=Op.create(OpType.Rz, [-op.params[1]]), qubits=qubits)
            self.apply_gate(op=Op.create(OpType.Rx, [op.params[0]]), qubits=qubits)
            self.apply_gate(op=Op.create(OpType.Rz, [op.params[1]]), qubits=qubits)

        elif op.type == OpType.ZZMax:
            self.apply_gate(op=Op.create(OpType.CX), qubits=qubits)
            self.apply_gate(op=Op.create(OpType.S), qubits=[qubits[1]])
            self.apply_gate(op=Op.create(OpType.CX), qubits=qubits)

        elif op.type == OpType.ZZPhase:
            for _ in range(int((op.params[0] % 2) // 0.5)):
                self.apply_gate(op=Op.create(OpType.ZZMax), qubits=qubits)

        else:
            raise NotImplementedError(
                f"{op} if clifford but is not supported. "
                "Please request the developers support this operation."
            )

    def pre_apply_pauli(self, pauli, qubit):
        mult_string, mult_coeff = pauli_string_mult(
            qubitpaulistring1=self.qubit_pauli_tensor.string,
            qubitpaulistring2=QubitPauliString(qubit=qubit, pauli=pauli),
        )
        mult_string_map = {
            self.qubit_index[qubit]: pauli for qubit, pauli in mult_string.map.items()
        }
        mult_string = QubitPauliString(map=mult_string_map)

        self.input_pauli_tensor = QubitPauliTensor(
            string=mult_string, coeff=self.qubit_pauli_tensor.coeff * mult_coeff
        )

        self.unitary_tableau = UnitaryTableau(nqb=len(self.qubit_list))

    def post_apply_pauli(self, pauli, qubit):
        mult_string, mult_coeff = pauli_string_mult(
            qubitpaulistring1=QubitPauliString(qubit=qubit, pauli=pauli),
            qubitpaulistring2=self.qubit_pauli_tensor.string,
        )
        mult_string_map = {
            self.qubit_index[qubit]: pauli for qubit, pauli in mult_string.map.items()
        }
        mult_string = QubitPauliString(map=mult_string_map)

        self.input_pauli_tensor = QubitPauliTensor(
            string=mult_string, coeff=self.qubit_pauli_tensor.coeff * mult_coeff
        )

        self.unitary_tableau = UnitaryTableau(nqb=len(self.qubit_list))

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
    def qubit_pauli_tensor(self):
        mislabled_qpt = self.unitary_tableau.get_row_product(
            paulis=self.input_pauli_tensor
        )
        correct_map = {
            qubit: mislabled_qpt.string.map.get(index, Pauli.I)
            for index, qubit in self.index_qubit.items()
        }
        return QubitPauliTensor(
            string=QubitPauliString(
                map=correct_map,
            ),
            coeff=mislabled_qpt.coeff,
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
        return cls(
            qpt=QubitPauliTensor(
                string=QubitPauliString(
                    qubits=qubit_list,
                    paulis=pauli_list,
                ),
                coeff=1,
            ),
        )

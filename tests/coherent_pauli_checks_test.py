from collections import Counter

import networkx as nx
import numpy as np
import numpy.random
import pytest
from pytket import Circuit, OpType
from pytket.circuit import Bit, CircBox, Qubit
from pytket.extensions.qiskit import AerBackend
from pytket.passes import DecomposeBoxes
from pytket.pauli import Pauli, QubitPauliString

from qermit import CircuitShots
from qermit.coherent_pauli_checks import (
    DeterministicXPauliSampler,
    DeterministicZPauliSampler,
    OptimalPauliSampler,
    PauliSampler,
    RandomPauliSampler,
    cpc_rebase_pass,
    gen_coherent_pauli_check_mitres,
)
from qermit.coherent_pauli_checks.box_clifford_subcircuits import (
    BoxClifford,
    circuit_to_graph,
    command_is_clifford,
    get_clifford_commands,
    get_clifford_subcircuits,
)
from qermit.coherent_pauli_checks.monochromatic_convex_subdag import (
    MonochromaticConvexSubDAG,
)
from qermit.noise_model import (
    Direction,
    ErrorDistribution,
    NoiseModel,
    PauliErrorTranspile,
    TranspilerBackend,
)
from qermit.noise_model.noise_model import Direction, LogicalErrorDistribution
from qermit.noise_model.qermit_pauli import QermitPauli
from qermit.postselection import PostselectMgr
from qermit.probabilistic_error_cancellation.cliff_circuit_gen import (
    random_clifford_circ,
)


def test_command_is_clifford():
    qubit_0 = Qubit(name="my_qubit_0", index=0)
    qubit_1 = Qubit(name="my_qubit_0", index=2)
    qubit_2 = Qubit(name="my_qubit_1", index=0)
    qubit_3 = Qubit(name="my_qubit_1", index=1)

    bit_0 = Bit(name="my_bit_0", index=0)
    bit_1 = Bit(name="my_bit_0", index=2)
    bit_2 = Bit(name="my_bit_1", index=0)

    circuit = Circuit()

    circuit.add_qubit(qubit_0)
    circuit.add_qubit(qubit_1)
    circuit.add_qubit(qubit_2)
    circuit.add_qubit(qubit_3)

    circuit.add_bit(bit_0)
    circuit.add_bit(bit_1)
    circuit.add_bit(bit_2)

    circuit.CZ(qubit_0, qubit_1)
    circuit.X(qubit_2, condition=bit_2)
    circuit.Y(qubit_3)
    circuit.PhasedX(0.5, 0.5, qubit_1)
    circuit.CX(qubit_2, qubit_0, condition=bit_1)
    circuit.add_c_and(arg0_in=bit_0, arg1_in=bit_2, arg_out=bit_1)
    circuit.Z(qubit_1, condition=bit_0)
    circuit.Rz(0.5, qubit_2)
    circuit.PhasedX(0.5, 0.5, qubit_2, condition=bit_0)
    circuit.Rz(0.5, qubit_2, condition=bit_2)
    circuit.H(qubit_1)
    circuit.Rz(0.55, qubit_2)
    circuit.PhasedX(0.5, 0.55, qubit_2)

    command_list = circuit.get_commands()

    assert command_list[0].op.type == OpType.CZ
    assert command_is_clifford(command_list[0])

    assert command_list[1].op.type == OpType.Conditional
    assert not command_is_clifford(command_list[1])

    assert command_list[2].op.type == OpType.Y
    assert command_is_clifford(command_list[2])

    assert command_list[3].op.type == OpType.Conditional
    assert not command_is_clifford(command_list[3])

    assert command_list[4].op.type == OpType.PhasedX
    assert command_is_clifford(command_list[4])

    assert command_list[5].op.type == OpType.ExplicitPredicate
    assert not command_is_clifford(command_list[5])

    assert command_list[6].op.type == OpType.Conditional
    assert not command_is_clifford(command_list[6])

    assert command_list[7].op.type == OpType.Rz
    assert command_is_clifford(command_list[7])

    assert command_list[8].op.type == OpType.H
    assert command_is_clifford(command_list[8])

    assert command_list[9].op.type == OpType.Conditional
    assert not command_is_clifford(command_list[9])

    assert command_list[10].op.type == OpType.Conditional
    assert not command_is_clifford(command_list[10])

    assert command_list[11].op.type == OpType.Rz
    assert not command_is_clifford(command_list[11])

    assert command_list[12].op.type == OpType.PhasedX
    assert not command_is_clifford(command_list[12])


def test_monochromatic_convex_subdag():
    dag = nx.DiGraph()
    dag.add_edges_from([(1, 2), (1, 3), (2, 4)])

    nx.draw(dag, with_labels=True)

    convex_subdag = MonochromaticConvexSubDAG(dag=dag, coloured_nodes=[1, 2])

    node_subdag = {1: 0, 2: 1, 3: 2, 4: 3}

    assert convex_subdag._can_merge(subdag_one=1, subdag_two=0, node_subdag=node_subdag)
    assert convex_subdag._can_merge(subdag_one=0, subdag_two=1, node_subdag=node_subdag)
    assert convex_subdag._can_merge(subdag_one=2, subdag_two=1, node_subdag=node_subdag)
    assert not convex_subdag._can_merge(
        subdag_one=0, subdag_two=3, node_subdag=node_subdag
    )

    assert convex_subdag._subdag_successors(
        subdag=0, node_subdag={1: 0, 2: 0, 3: 0, 4: 3}
    ) == [4]
    assert (
        convex_subdag._subdag_successors(subdag=3, node_subdag={1: 0, 2: 0, 3: 0, 4: 3})
        == []
    )
    assert (
        convex_subdag._subdag_predecessors(
            subdag=0, node_subdag={1: 0, 2: 0, 3: 0, 4: 3}
        )
        == []
    )
    assert convex_subdag._subdag_predecessors(
        subdag=3, node_subdag={1: 0, 2: 0, 3: 0, 4: 3}
    ) == [2]
    assert convex_subdag._can_merge(
        subdag_one=0, subdag_two=3, node_subdag={1: 0, 2: 0, 3: 0, 4: 3}
    )

    assert convex_subdag.greedy_merge() == {1: 0, 2: 0}

    convex_subdag = MonochromaticConvexSubDAG(dag=dag, coloured_nodes=[1, 4])
    assert convex_subdag.greedy_merge() == {1: 0, 4: 1}

    convex_subdag = MonochromaticConvexSubDAG(dag=dag, coloured_nodes=[2, 3])
    assert convex_subdag.greedy_merge() == {2: 0, 3: 0}


def test_two_clifford_boxes() -> None:
    cx_error_distribution = ErrorDistribution(
        rng=np.random.default_rng(),
        distribution={
            (Pauli.X, Pauli.I): 0.1,
        },
    )

    noise_model = NoiseModel(
        noise_model={
            OpType.CX: cx_error_distribution,
        },
    )

    transpiler = PauliErrorTranspile(noise_model=noise_model)
    backend = TranspilerBackend(transpiler=transpiler)

    cliff_circ = Circuit(3)

    cliff_subcirc = Circuit(3, name="Clifford Subcircuit").CX(0, 1).CX(1, 2)
    cliff_circ.add_circbox(circbox=CircBox(circ=cliff_subcirc), args=cliff_circ.qubits)

    cliff_subcirc = Circuit(3, name="Clifford Subcircuit").CX(1, 2).CX(1, 0)
    cliff_circ.add_circbox(circbox=CircBox(circ=cliff_subcirc), args=cliff_circ.qubits)

    cliff_circ.measure_all()

    pauli_sampler = OptimalPauliSampler(
        noise_model=noise_model,
        n_checks=2,
    )

    pauli_check_circuit, postselect_cbits = pauli_sampler.add_pauli_checks_to_circbox(
        circuit=cliff_circ
    )

    postselect_mgr = PostselectMgr(
        compute_cbits=cliff_circ.bits,
        postselect_cbits=list(postselect_cbits),
    )

    DecomposeBoxes().apply(pauli_check_circuit)
    result = backend.run_circuit(pauli_check_circuit, 1000)
    postselect_result = postselect_mgr.postselect_result(result)
    postselect_result.get_counts()

    assert list(postselect_result.get_counts().keys()) == [(0, 0, 0)]


def test_coherent_pauli_checks_mitres() -> None:
    error_distribution = ErrorDistribution(
        rng=np.random.default_rng(),
        distribution={
            (Pauli.X, Pauli.I): 0.1,
        },
    )

    noise_model = NoiseModel(
        noise_model={OpType.CX: error_distribution},
    )

    transpiler = PauliErrorTranspile(noise_model=noise_model)
    backend = TranspilerBackend(transpiler=transpiler)

    cliff_circ = Circuit(3).CX(0, 1).CX(1, 2).measure_all()

    pauli_sampler = OptimalPauliSampler(
        noise_model=noise_model,
        n_checks=2,
    )

    mitres = gen_coherent_pauli_check_mitres(
        backend=backend, pauli_sampler=pauli_sampler
    )

    result_list = mitres.run([CircuitShots(Circuit=cliff_circ, Shots=1000)])

    assert list(result_list[0].get_counts().keys()) == [(0, 0, 0)]


def test_logical_error_coherent_pauli_check_workflow():
    cliff_sub_circ = Circuit(3, name="Clifford Subcircuit").CX(0, 1).Z(1).CZ(2, 1)

    error_distribution = ErrorDistribution(
        distribution={(Pauli.X, Pauli.X): 0.1}, rng=np.random.default_rng(seed=0)
    )
    noise_model = NoiseModel(
        noise_model={OpType.CZ: error_distribution},
    )

    pauli_sampler = OptimalPauliSampler(noise_model=noise_model, n_checks=1)
    assert pauli_sampler.sample(circ=cliff_sub_circ)[0].qubit_pauli_string == (
        QubitPauliString(
            qubits=[Qubit(0), Qubit(1), Qubit(2)], paulis=[Pauli.I, Pauli.I, Pauli.Z]
        ),
        1,
    )

    cliff_circ = Circuit()
    cliff_circ.add_q_register(name="my_reg", size=3)
    cliff_circ.add_circbox(CircBox(circ=cliff_sub_circ), cliff_circ.qubits)

    checked_cliff_circ, _ = pauli_sampler.add_pauli_checks_to_circbox(
        circuit=cliff_circ,
    )

    DecomposeBoxes().apply(checked_cliff_circ)

    n_counts = 10000
    logical_error_counter = noise_model.counter_propagate(
        cliff_circ=checked_cliff_circ,
        n_counts=n_counts,
        direction=Direction.forward,
    )
    logical_error_distribution = LogicalErrorDistribution(
        pauli_error_counter=logical_error_counter,
        total=n_counts,
    )
    logical_error_distribution_dict = logical_error_distribution.distribution

    qubits = [
        Qubit(name="ancilla", index=0),
        Qubit(name="my_reg", index=0),
        Qubit(name="my_reg", index=1),
        Qubit(name="my_reg", index=2),
    ]
    tol = 0.01

    assert (
        abs(
            logical_error_distribution_dict[
                QubitPauliString(
                    qubits=qubits, paulis=[Pauli.Z, Pauli.I, Pauli.I, Pauli.X]
                )
            ]
            - 0.081
        )
        < tol
    )
    assert (
        abs(
            logical_error_distribution_dict[
                QubitPauliString(
                    qubits=qubits, paulis=[Pauli.X, Pauli.I, Pauli.X, Pauli.X]
                )
            ]
            - 0.081
        )
        < tol
    )
    assert (
        abs(
            logical_error_distribution_dict[
                QubitPauliString(
                    qubits=qubits, paulis=[Pauli.Y, Pauli.I, Pauli.Z, Pauli.Y]
                )
            ]
            - 0.081
        )
        < tol
    )

    assert (
        abs(
            logical_error_distribution_dict[
                QubitPauliString(
                    qubits=qubits, paulis=[Pauli.Y, Pauli.I, Pauli.X, Pauli.I]
                )
            ]
            - 0.009
        )
        < tol
    )

    assert (
        abs(
            logical_error_distribution_dict[
                QubitPauliString(
                    qubits=qubits, paulis=[Pauli.X, Pauli.I, Pauli.Z, Pauli.Z]
                )
            ]
            - 0.009
        )
        < tol
    )

    assert (
        abs(
            logical_error_distribution_dict[
                QubitPauliString(
                    qubits=qubits, paulis=[Pauli.Z, Pauli.I, Pauli.Y, Pauli.Z]
                )
            ]
            - 0.009
        )
        < tol
    )

    assert (
        abs(
            logical_error_distribution_dict[
                QubitPauliString(
                    qubits=qubits, paulis=[Pauli.I, Pauli.I, Pauli.Y, Pauli.Y]
                )
            ]
            - 0.001
        )
        < tol
    )


def test_decompose_clifford_subcircuit_box():
    cx_circ = Circuit(3)
    cx_circ.CX(0, 1).CX(0, 2)

    cx_circbox = CircBox(circ=cx_circ)

    zzmax_circ = Circuit()
    zzmax_circ.add_q_register(name="my_reg", size=4)
    qubits = zzmax_circ.qubits

    zzmax_circ.ZZMax(qubits[0], qubits[1]).ZZMax(qubits[2], qubits[3])
    zzmax_circ.add_circbox(cx_circbox, [qubits[1], qubits[0], qubits[3]])

    dag_circ = PauliSampler._decompose_clifford_subcircuit_box(
        zzmax_circ.get_commands()[2]
    )

    ideal_circ = Circuit()
    ideal_circ.add_qubit(qubits[0])
    ideal_circ.add_qubit(qubits[1])
    ideal_circ.add_qubit(qubits[3])
    ideal_circ.CX(qubits[1], qubits[0])
    ideal_circ.CX(qubits[1], qubits[3])

    assert dag_circ == ideal_circ


def test_get_clifford_subcircuits():
    circ = Circuit(3).CZ(0, 1).H(1).Z(1).CZ(1, 0)
    dag, node_command = circuit_to_graph(circuit=circ)
    clifford_nodes = get_clifford_commands(node_command)
    assert get_clifford_subcircuits(dag=dag, clifford_nodes=clifford_nodes) == [
        0,
        0,
        0,
        0,
    ]

    circ = Circuit(3).CZ(1, 2).H(2).Z(1).CZ(0, 1).H(1).CZ(1, 0).Z(1).CZ(1, 2)
    dag, node_command = circuit_to_graph(circuit=circ)
    clifford_nodes = get_clifford_commands(node_command)
    assert get_clifford_subcircuits(dag=dag, clifford_nodes=clifford_nodes) == [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]

    circ = (
        Circuit(3)
        .CZ(1, 2)
        .Rz(0.1, 1)
        .H(2)
        .Z(1)
        .CZ(0, 1)
        .Rz(0.1, 1)
        .CZ(1, 0)
        .Z(1)
        .CZ(1, 2)
    )
    dag, node_command = circuit_to_graph(circuit=circ)
    clifford_nodes = get_clifford_commands(node_command)
    assert get_clifford_subcircuits(dag=dag, clifford_nodes=clifford_nodes) == [
        0,
        1,
        0,
        2,
        2,
        3,
        4,
        4,
        4,
    ]


def test_add_pauli_checks():
    boxed_circ = Circuit(3).H(1).CX(1, 0)
    cpc_rebase_pass.apply(boxed_circ)
    BoxClifford().apply(boxed_circ)

    circuit, _ = DeterministicZPauliSampler().add_pauli_checks_to_circbox(
        circuit=boxed_circ,
    )
    DecomposeBoxes().apply(circuit)

    ideal_circ = Circuit(3)

    ancilla_0 = Qubit(name="ancilla", index=0)
    ideal_circ.add_qubit(ancilla_0)

    ancilla_measure_0 = Bit(name="ancilla_measure", index=0)
    ideal_circ.add_bit(id=ancilla_measure_0)

    ideal_circ.add_barrier([ideal_circ.qubits[2], ideal_circ.qubits[1], ancilla_0])
    ideal_circ.H(ancilla_0, opgroup="ancilla superposition")
    ideal_circ.CZ(
        control_qubit=ancilla_0,
        target_qubit=ideal_circ.qubits[1],
        opgroup="pauli check",
    )
    ideal_circ.CZ(
        control_qubit=ancilla_0,
        target_qubit=ideal_circ.qubits[2],
        opgroup="pauli check",
    )
    ideal_circ.add_barrier([ideal_circ.qubits[2], ideal_circ.qubits[1], ancilla_0])

    ideal_circ.H(ideal_circ.qubits[2])

    ideal_circ.H(ideal_circ.qubits[1])
    ideal_circ.CZ(control_qubit=ideal_circ.qubits[2], target_qubit=ideal_circ.qubits[1])
    ideal_circ.H(ideal_circ.qubits[1])

    ideal_circ.add_barrier([ideal_circ.qubits[2], ideal_circ.qubits[1], ancilla_0])
    ideal_circ.CZ(
        control_qubit=ancilla_0,
        target_qubit=ideal_circ.qubits[1],
        opgroup="pauli check",
    )
    ideal_circ.CX(
        control_qubit=ancilla_0,
        target_qubit=ideal_circ.qubits[1],
        opgroup="pauli check",
    )
    ideal_circ.CZ(
        control_qubit=ancilla_0,
        target_qubit=ideal_circ.qubits[2],
        opgroup="pauli check",
    )
    ideal_circ.CX(
        control_qubit=ancilla_0,
        target_qubit=ideal_circ.qubits[2],
        opgroup="pauli check",
    )
    ideal_circ.H(ancilla_0, opgroup="ancilla superposition")
    ideal_circ.add_barrier([ideal_circ.qubits[2], ideal_circ.qubits[1], ancilla_0])

    ideal_circ.Measure(ancilla_0, ancilla_measure_0)

    assert ideal_circ == circuit

    boxed_circ = Circuit(2).H(0).CX(1, 0).X(1).CX(1, 0)
    cpc_rebase_pass.apply(boxed_circ)
    BoxClifford().apply(boxed_circ)

    circuit, _ = DeterministicZPauliSampler().add_pauli_checks_to_circbox(
        circuit=boxed_circ,
    )
    DecomposeBoxes().apply(circuit)

    ideal_circ = Circuit(2)

    ancilla = Qubit(name="ancilla", index=0)
    ancilla_measure = Bit(name="ancilla_measure", index=0)

    ideal_circ.add_qubit(ancilla)
    ideal_circ.add_bit(id=ancilla_measure)

    ideal_circ.add_barrier([ideal_circ.qubits[2], ideal_circ.qubits[1], ancilla])
    ideal_circ.H(ancilla, opgroup="ancilla superposition")
    ideal_circ.CZ(
        control_qubit=ancilla, target_qubit=ideal_circ.qubits[1], opgroup="pauli check"
    )
    ideal_circ.CZ(
        control_qubit=ancilla, target_qubit=ideal_circ.qubits[2], opgroup="pauli check"
    )
    ideal_circ.add_barrier([ideal_circ.qubits[2], ideal_circ.qubits[1], ancilla])

    ideal_circ.H(ideal_circ.qubits[1])
    ideal_circ.H(ideal_circ.qubits[1])
    ideal_circ.CZ(control_qubit=ideal_circ.qubits[2], target_qubit=ideal_circ.qubits[1])
    ideal_circ.H(ideal_circ.qubits[1])
    ideal_circ.H(ideal_circ.qubits[1])
    ideal_circ.X(ideal_circ.qubits[2])
    ideal_circ.CZ(control_qubit=ideal_circ.qubits[2], target_qubit=ideal_circ.qubits[1])
    ideal_circ.H(ideal_circ.qubits[1])

    ideal_circ.add_barrier([ideal_circ.qubits[2], ideal_circ.qubits[1], ancilla])
    ideal_circ.CX(
        control_qubit=ancilla, target_qubit=ideal_circ.qubits[1], opgroup="pauli check"
    )
    ideal_circ.CZ(
        control_qubit=ancilla, target_qubit=ideal_circ.qubits[2], opgroup="pauli check"
    )
    ideal_circ.S(ancilla, opgroup="phase correction")
    ideal_circ.S(ancilla, opgroup="phase correction")
    ideal_circ.H(ancilla, opgroup="ancilla superposition")
    ideal_circ.add_barrier([ideal_circ.qubits[2], ideal_circ.qubits[1], ancilla])

    ideal_circ.Measure(ancilla, ancilla_measure)

    assert ideal_circ == circuit


def test_simple_example():
    clifford_circuit = Circuit(3).CZ(0, 1).X(2).X(0).CZ(0, 2).CZ(1, 2)
    dag, node_command = circuit_to_graph(circuit=clifford_circuit)
    clifford_nodes = get_clifford_commands(node_command)
    assert get_clifford_subcircuits(dag=dag, clifford_nodes=clifford_nodes) == [
        0,
        0,
        0,
        0,
        0,
    ]


def test_5q_random_clifford():
    rng = numpy.random.default_rng(seed=0)
    boxed_clifford_circuit = random_clifford_circ(n_qubits=5, rng=rng)
    cpc_rebase_pass.apply(boxed_clifford_circuit)
    BoxClifford().apply(boxed_clifford_circuit)
    pauli_sampler = RandomPauliSampler(rng=rng, n_checks=2)
    pauli_sampler.add_pauli_checks_to_circbox(circuit=boxed_clifford_circuit)


def test_CZ_circuit_with_phase():
    # This test is a case where the pauli circuit to be controlled has a
    # global phase which needs to be bumped to the control.

    boxed_original_circuit = Circuit(2).CZ(0, 1).measure_all()
    BoxClifford().apply(boxed_original_circuit)
    pauli_sampler = DeterministicXPauliSampler()
    pauli_checks_circuit, _ = pauli_sampler.add_pauli_checks_to_circbox(
        circuit=boxed_original_circuit,
    )
    DecomposeBoxes().apply(pauli_checks_circuit)

    ideal_circ = Circuit(2, 2)
    comp_qubits = ideal_circ.qubits
    comp_bits = ideal_circ.bits

    ancilla = Qubit(name="ancilla", index=0)
    ancilla_measure = Bit(name="ancilla_measure", index=0)

    ideal_circ.add_qubit(ancilla)
    ideal_circ.add_bit(id=ancilla_measure)

    ideal_circ.add_barrier([*list(reversed(comp_qubits)), ancilla])
    ideal_circ.H(ancilla, opgroup="ancilla superposition")
    ideal_circ.CX(ancilla, comp_qubits[0], opgroup="pauli check")
    ideal_circ.CX(ancilla, comp_qubits[1], opgroup="pauli check")
    ideal_circ.add_barrier([*list(reversed(comp_qubits)), ancilla])
    ideal_circ.CZ(comp_qubits[0], comp_qubits[1])
    ideal_circ.add_barrier([*list(reversed(comp_qubits)), ancilla])
    ideal_circ.CZ(ancilla, comp_qubits[0], opgroup="pauli check")
    ideal_circ.CX(ancilla, comp_qubits[0], opgroup="pauli check")
    ideal_circ.CZ(ancilla, comp_qubits[1], opgroup="pauli check")
    ideal_circ.CX(ancilla, comp_qubits[1], opgroup="pauli check")
    ideal_circ.S(ancilla, opgroup="phase correction")
    ideal_circ.S(ancilla, opgroup="phase correction")
    ideal_circ.H(ancilla, opgroup="ancilla superposition")
    ideal_circ.add_barrier([*list(reversed(comp_qubits)), ancilla])
    ideal_circ.Measure(comp_qubits[0], comp_bits[0])
    ideal_circ.Measure(comp_qubits[1], comp_bits[1])
    ideal_circ.Measure(ancilla, ancilla_measure)

    assert pauli_checks_circuit == ideal_circ


def test_to_clifford_subcircuits():
    orig_circuit = (
        Circuit(3)
        .CZ(1, 2)
        .Rz(0.1, 1)
        .H(2)
        .Z(1)
        .CZ(0, 1)
        .Rz(0.1, 1)
        .CZ(1, 0)
        .Z(1)
        .CZ(1, 2)
    )
    # dag_circuit = QermitDAGCircuit(orig_circuit)
    # clifford_box_circuit = dag_circuit.to_clifford_subcircuit_boxes()
    clifford_box_circuit = orig_circuit.copy()
    BoxClifford().apply(clifford_box_circuit)
    DecomposeBoxes().apply(clifford_box_circuit)
    assert clifford_box_circuit == orig_circuit


def test_optimal_pauli_sampler():
    # TODO: add a measure and barrier to this circuit, just to check
    boxed_cliff_circ = Circuit()
    boxed_cliff_circ.add_q_register(name="my_reg", size=3)
    qubits = boxed_cliff_circ.qubits
    boxed_cliff_circ.CZ(qubits[0], qubits[1]).CZ(qubits[1], qubits[2])

    error_distribution_dict = {}
    error_distribution_dict[(Pauli.X, Pauli.I)] = 0.3
    error_distribution_dict[(Pauli.I, Pauli.X)] = 0.7

    error_distribution = ErrorDistribution(
        error_distribution_dict, rng=numpy.random.default_rng(seed=0)
    )
    noise_model = NoiseModel(
        noise_model={OpType.CZ: error_distribution},
    )

    pauli_sampler = OptimalPauliSampler(
        noise_model=noise_model,
        n_checks=1,
    )
    stab = pauli_sampler.sample(
        circ=boxed_cliff_circ,
    )

    assert stab[0] == QermitPauli(
        Z_list=[0, 0, 1],
        X_list=[0, 0, 1],
        qubit_list=qubits,
        phase=1,
    )

    # TODO: an assert is needed for this last part

    pauli_sampler = OptimalPauliSampler(
        noise_model=noise_model,
        n_checks=2,
    )
    BoxClifford().apply(boxed_cliff_circ)
    pauli_sampler.add_pauli_checks_to_circbox(circuit=boxed_cliff_circ)


def test_add_ZX_pauli_checks_to_S():
    boxed_cliff_circ = Circuit()
    boxed_cliff_circ.add_q_register(name="my_reg", size=1)
    qubits = boxed_cliff_circ.qubits
    boxed_cliff_circ.S(qubits[0])
    boxed_cliff_circ.measure_all()

    class DeterministicPauliSampler(PauliSampler):
        def sample(self, circ, **kwargs):
            qubit_list = circ.qubits
            return [
                QermitPauli(
                    Z_list=[1],
                    X_list=[1],
                    qubit_list=qubit_list,
                )
            ]

    BoxClifford().apply(boxed_cliff_circ)
    pauli_sampler = DeterministicPauliSampler()
    pauli_check_circ, _ = pauli_sampler.add_pauli_checks_to_circbox(
        circuit=boxed_cliff_circ,
    )

    DecomposeBoxes().apply(pauli_check_circ)

    ideal_circ = Circuit()

    ancilla = Qubit(name="ancilla", index=0)
    comp = Qubit(name="my_reg", index=0)
    ancilla_measure = Bit(name="ancilla_measure", index=0)
    comp_measure = Bit(name="c", index=0)

    ideal_circ.add_qubit(ancilla)
    ideal_circ.add_qubit(comp)

    ideal_circ.add_bit(id=ancilla_measure)
    ideal_circ.add_bit(id=comp_measure)

    ideal_circ.add_barrier([comp, ancilla])
    ideal_circ.H(ancilla, opgroup="ancilla superposition")
    ideal_circ.CZ(ancilla, comp, opgroup="pauli check")
    ideal_circ.CX(ancilla, comp, opgroup="pauli check")
    ideal_circ.add_barrier([comp, ancilla])
    ideal_circ.S(comp)
    ideal_circ.add_barrier([comp, ancilla])
    ideal_circ.CX(ancilla, comp, opgroup="pauli check")
    ideal_circ.S(ancilla, opgroup="phase correction")
    ideal_circ.S(ancilla, opgroup="phase correction")
    ideal_circ.S(ancilla, opgroup="phase correction")
    ideal_circ.H(ancilla, opgroup="ancilla superposition")
    ideal_circ.add_barrier([comp, ancilla])
    ideal_circ.Measure(ancilla, ancilla_measure)
    ideal_circ.Measure(comp, comp_measure)

    assert ideal_circ == pauli_check_circ

    backend = AerBackend()
    backend.rebase_pass().apply(pauli_check_circ)
    result = backend.run_circuit(pauli_check_circ, n_shots=100)
    counts = result.get_counts()

    assert list(counts.keys()) == [(0, 0)]


@pytest.mark.high_compute
def test_error_sampler():
    n_shots = 1000

    distribution = {
        (Pauli.Z, Pauli.Z): 0.001,
        (Pauli.I, Pauli.Z): 0.01,
        (Pauli.Z, Pauli.I): 0.01,
    }

    error_distribution = ErrorDistribution(
        distribution=distribution,
        rng=numpy.random.default_rng(seed=1),
    )

    noise_model = NoiseModel(
        noise_model={OpType.ZZMax: error_distribution},
    )

    cliff_circ = Circuit(2, name="Clifford Subcircuit").H(0)
    for _ in range(32):
        cliff_circ.ZZMax(0, 1)
    cliff_circ = cliff_circ.H(0)

    circuit = Circuit()
    circuit.add_q_register(
        name="my_reg",
        size=2,
    )
    circuit.H(circuit.qubits[0])
    circuit.add_circbox(
        circbox=CircBox(cliff_circ), args=list(reversed(circuit.qubits))
    )
    circuit.H(circuit.qubits[0])
    circuit.measure_all()

    pauli_sampler = OptimalPauliSampler(
        noise_model=noise_model,
        n_checks=1,
    )
    checked_circuit, _ = pauli_sampler.add_pauli_checks_to_circbox(
        circuit=circuit,
    )

    transpiler = PauliErrorTranspile(noise_model=noise_model)
    backend = TranspilerBackend(transpiler=transpiler)

    DecomposeBoxes().apply(checked_circuit)

    postselect_mgr = PostselectMgr(
        compute_cbits=checked_circuit.bits[1:],
        postselect_cbits=checked_circuit.bits[:1],
    )

    result = backend.run_circuit(checked_circuit, n_shots=n_shots)
    postselected_result = postselect_mgr.postselect_result(result=result)
    # These asserts reveal that the least dominant error is not detected
    assert result.get_counts() == Counter(
        {
            (0, 0, 0): 551,
            (1, 1, 0): 196,
            (1, 0, 1): 188,
            (0, 1, 1): 65,
        }
    )
    assert postselected_result.get_counts() == Counter({(0, 0): 551, (1, 1): 65})

    error_counter = noise_model.counter_propagate(
        cliff_circ=checked_circuit,
        n_counts=n_shots,
        direction=Direction.forward,
    )
    ideal = Counter(
        {
            QermitPauli(
                Z_list=[0, 0, 0],
                X_list=[1, 1, 0],
                qubit_list=[
                    Qubit(name="ancilla", index=0),
                    Qubit(name="my_reg", index=0),
                    Qubit(name="my_reg", index=1),
                ],
            ): 190,
            QermitPauli(
                Z_list=[0, 0, 0],
                X_list=[1, 0, 1],
                qubit_list=[
                    Qubit(name="ancilla", index=0),
                    Qubit(name="my_reg", index=0),
                    Qubit(name="my_reg", index=1),
                ],
            ): 164,
            QermitPauli(
                Z_list=[0, 0, 0],
                X_list=[0, 1, 1],
                qubit_list=[
                    Qubit(name="ancilla", index=0),
                    Qubit(name="my_reg", index=0),
                    Qubit(name="my_reg", index=1),
                ],
            ): 81,
        }
    )
    # We see here again that the IXX error will be undetected.
    assert error_counter == ideal

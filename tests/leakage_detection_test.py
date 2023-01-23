# Copyright 2019-2022 Cambridge Quantum Computing
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from qermit.leakage_detection import (  # type: ignore
    gen_Leakage_Detection_MitRes,
    postselection_circuits_task_gen,
    postselection_results_task_gen,
)
from qermit import CircuitShots  # type: ignore
from qermit.taskgraph.mitex import gen_compiled_MitRes  # type: ignore
from pytket import Circuit, Qubit, Bit, OpType
from pytket.extensions.qiskit import AerBackend  # type: ignore
import pytest  # type: ignore


def test_postselection_circuits_1qb_task_gen() -> None:
    # with enough spare qubits
    circuits_mittask = postselection_circuits_task_gen(2)
    experiment_circuit = Circuit(1, 1).H(0).Measure(0, 0)
    output = circuits_mittask(([CircuitShots(experiment_circuit, 10)],))
    comparison_circuit = Circuit(1, 1)
    lg_qb = Qubit("leakage_gadget_qubit", 0)
    lg_b = Bit("leakage_gadget_bit", 0)
    comparison_circuit.add_qubit(lg_qb)
    comparison_circuit.add_bit(lg_b)
    comparison_circuit.X(lg_qb)
    comparison_circuit.H(0)
    comparison_circuit.add_barrier([Qubit(0), lg_qb])
    comparison_circuit.H(lg_qb)
    comparison_circuit.ZZMax(lg_qb, Qubit(0))
    comparison_circuit.add_barrier([Qubit(0), lg_qb])
    comparison_circuit.ZZMax(lg_qb, Qubit(0))
    comparison_circuit.H(lg_qb)
    comparison_circuit.Z(0)
    comparison_circuit.add_barrier([Qubit(0), lg_qb])
    comparison_circuit.Measure(0, 0)
    comparison_circuit.Measure(lg_qb, lg_b)
    comparison_circuit.add_gate(OpType.Reset, [lg_qb])
    assert comparison_circuit == output[0][0][0]


def test_postselection_circuits_2qb_2_spare_task_gen() -> None:
    circuits_mittask = postselection_circuits_task_gen(4)
    experiment_circuit = Circuit(2, 2).H(0).CZ(0, 1).Measure(0, 0).Measure(1, 1)
    output = circuits_mittask(([CircuitShots(experiment_circuit, 10)],))
    comparison_circuit = Circuit(2, 2)
    lg_qb0 = Qubit("leakage_gadget_qubit", 0)
    lg_b0 = Bit("leakage_gadget_bit", 0)
    lg_qb1 = Qubit("leakage_gadget_qubit", 1)
    lg_b1 = Bit("leakage_gadget_bit", 1)
    comparison_circuit.add_qubit(lg_qb0)
    comparison_circuit.add_bit(lg_b0)
    comparison_circuit.add_qubit(lg_qb1)
    comparison_circuit.add_bit(lg_b1)

    comparison_circuit.X(lg_qb0)
    comparison_circuit.X(lg_qb1)
    comparison_circuit.H(0)
    comparison_circuit.CZ(0, 1)
    comparison_circuit.add_barrier([Qubit(0), lg_qb0])
    comparison_circuit.add_barrier([Qubit(1), lg_qb1])
    comparison_circuit.H(lg_qb0)
    comparison_circuit.H(lg_qb1)
    comparison_circuit.ZZMax(lg_qb0, Qubit(0))
    comparison_circuit.ZZMax(lg_qb1, Qubit(1))
    comparison_circuit.add_barrier([Qubit(0), lg_qb0])
    comparison_circuit.add_barrier([Qubit(1), lg_qb1])
    comparison_circuit.ZZMax(lg_qb0, Qubit(0))
    comparison_circuit.ZZMax(lg_qb1, Qubit(1))
    comparison_circuit.H(lg_qb0)
    comparison_circuit.H(lg_qb1)
    comparison_circuit.Z(0)
    comparison_circuit.Z(1)
    comparison_circuit.add_barrier([Qubit(0), lg_qb0])
    comparison_circuit.add_barrier([Qubit(1), lg_qb1])
    comparison_circuit.Measure(0, 0)
    comparison_circuit.Measure(1, 1)
    comparison_circuit.Measure(lg_qb0, lg_b0)
    comparison_circuit.Measure(lg_qb1, lg_b1)
    comparison_circuit.add_gate(OpType.Reset, [lg_qb0])
    comparison_circuit.add_gate(OpType.Reset, [lg_qb1])
    assert comparison_circuit == output[0][0][0]


def test_postselection_circuits_2qb_1_spare_task_gen() -> None:
    circuits_mittask = postselection_circuits_task_gen(3)
    experiment_circuit = Circuit(2, 2).H(0).CZ(0, 1).Measure(0, 0).Measure(1, 1)
    output = circuits_mittask(([CircuitShots(experiment_circuit, 10)],))

    comparison_circuit = Circuit(2, 2)
    lg_qb = Qubit("leakage_gadget_qubit", 0)
    lg_b0 = Bit("leakage_gadget_bit", 0)
    lg_b1 = Bit("leakage_gadget_bit", 1)
    comparison_circuit.add_qubit(lg_qb)
    comparison_circuit.add_bit(lg_b0)
    comparison_circuit.add_bit(lg_b1)

    comparison_circuit.X(lg_qb)
    comparison_circuit.H(0)
    comparison_circuit.CZ(0, 1)
    comparison_circuit.add_barrier([Qubit(0), lg_qb])
    comparison_circuit.H(lg_qb)
    comparison_circuit.ZZMax(lg_qb, Qubit(0))
    comparison_circuit.add_barrier([Qubit(0), lg_qb])
    comparison_circuit.ZZMax(lg_qb, Qubit(0))
    comparison_circuit.H(lg_qb)
    comparison_circuit.Z(0)
    comparison_circuit.add_barrier([Qubit(0), lg_qb])
    comparison_circuit.Measure(0, 0)
    comparison_circuit.Measure(lg_qb, lg_b0)
    comparison_circuit.add_gate(OpType.Reset, [lg_qb])
    comparison_circuit.X(lg_qb)
    comparison_circuit.add_barrier([Qubit(1), lg_qb])
    comparison_circuit.H(lg_qb)
    comparison_circuit.ZZMax(lg_qb, Qubit(1))
    comparison_circuit.add_barrier([Qubit(1), lg_qb])
    comparison_circuit.ZZMax(lg_qb, Qubit(1))
    comparison_circuit.H(lg_qb)
    comparison_circuit.Z(1)
    comparison_circuit.add_barrier([Qubit(1), lg_qb])
    comparison_circuit.Measure(1, 1)
    comparison_circuit.Measure(lg_qb, lg_b1)
    comparison_circuit.add_gate(OpType.Reset, [lg_qb])

    assert comparison_circuit == output[0][0][0]


def test_postselection_discard() -> None:
    circuits_mittask = postselection_circuits_task_gen(2)
    experiment_circuit = Circuit(1, 1).H(0).Measure(0, 0)
    output = circuits_mittask(([CircuitShots(experiment_circuit, 10)],))
    lg_qb = Qubit("leakage_gadget_qubit", 0)
    c = Circuit(0)
    c.add_qubit(lg_qb)
    c.H(lg_qb)
    c.append(output[0][0][0])
    b = AerBackend()
    handle = b.process_circuit(b.get_compiled_circuit(c), 100)
    result = b.get_result(handle)
    counts = result.get_counts()
    assert (0, 0) in counts
    assert (0, 1) in counts
    assert (1, 0) in counts
    assert (1, 1) in counts
    postselection_task = postselection_results_task_gen()
    postselected_results = postselection_task(
        (
            (
                [result],
                [[Bit("leakage_gadget_bit", 0)]],
            )
        )
    )
    postselected_counts = postselected_results[0][0].get_counts()
    # just confirming that the result object returned has smaller width and fewer shots
    assert (0,) in postselected_counts
    assert (1,) in postselected_counts
    assert postselected_counts[(0,)] + postselected_counts[(1,)] < 100


def test_gen_Leakage_Detection_MitRes() -> None:
    b = AerBackend()
    ld_mitres = gen_Leakage_Detection_MitRes(b, mitres=gen_compiled_MitRes(b))
    experiment_circuit = Circuit(2).H(0).CX(0, 1).measure_all()

    res = ld_mitres.run([(b.get_compiled_circuit(experiment_circuit), 100)])
    counts = res[0].get_counts()
    print(counts.values())
    print(sum(counts.values()))
    assert sum(counts.values()) == 100


if __name__ == "__main__":
    test_postselection_circuits_1qb_task_gen()
    test_postselection_circuits_2qb_2_spare_task_gen()
    test_postselection_circuits_2qb_1_spare_task_gen()
    test_postselection_discard()
    test_gen_Leakage_Detection_MitRes()

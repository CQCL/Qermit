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


from qermit.postselection import (
    gen_Postselection_MitRes,
    postselection_results_task_gen,
    postselection_circuits_task_gen,
)
from qermit import CircuitShots
from pytket import Circuit, Qubit, Bit
from pytket.extensions.qiskit import AerBackend  # type: ignore


def test_postselection_circuits_task_gen():
    postselection_circuit = Circuit()
    ancilla_bit = Bit("ancilla_bit", 0)
    logical_qubit = Qubit("logical", 0)
    ancilla_qubit = Qubit("ancilla_qubit", 1)
    postselection_circuit.add_qubit(logical_qubit)
    postselection_circuit.add_qubit(ancilla_qubit)
    postselection_circuit.add_bit(ancilla_bit)
    postselection_circuit.CX(logical_qubit, ancilla_qubit)
    postselection_circuit.Measure(ancilla_qubit, ancilla_bit)

    circuits_mittask = postselection_circuits_task_gen(
        postselection_circuit, [logical_qubit], [(ancilla_qubit, ancilla_bit)]
    )
    experiment_circuit = Circuit(1, 1).H(0).Measure(0, 0)
    output = circuits_mittask(([CircuitShots(experiment_circuit, 10)],))
    comparison_circuit = Circuit(1, 1).H(0)
    comparison_circuit.add_qubit(ancilla_qubit)
    comparison_circuit.add_bit(ancilla_bit)
    comparison_circuit.CX(Qubit(0), ancilla_qubit).Measure(
        ancilla_qubit, ancilla_bit
    ).Measure(Qubit(0), Bit(0))
    assert comparison_circuit == output[0][0].Circuit


def test_gen_Postselection_MitRes():
    postselection_circuit = Circuit()
    ancilla_bit = Bit("ancilla_bit", 0)
    logical_qubit = Qubit("logical", 0)
    ancilla_qubit = Qubit("ancilla_qubit", 1)
    postselection_circuit.add_qubit(logical_qubit)
    postselection_circuit.add_qubit(ancilla_qubit)
    postselection_circuit.add_bit(ancilla_bit)
    postselection_circuit.CX(logical_qubit, ancilla_qubit)
    postselection_circuit.Measure(ancilla_qubit, ancilla_bit)

    backend = AerBackend()
    combo = (1,)
    banned_results = ((ancilla_bit,), {combo})
    ps_mr = gen_Postselection_MitRes(
        backend=backend,
        postselection_circuit=postselection_circuit,
        postselection_ancillas=[(ancilla_qubit, ancilla_bit)],
        postselection_logical_qubits=[logical_qubit],
        banned_results=banned_results,
    )

    experiment_circuit = Circuit(1, 1).H(0).Measure(0, 0)
    res = ps_mr.run([(experiment_circuit, 100)])
    counts_keys = res[0].get_counts().keys()
    assert (0, 0) in counts_keys
    assert (1, 1) not in counts_keys


if __name__ == "__main__":
    test_postselection_circuits_task_gen()
    test_gen_Postselection_MitRes()

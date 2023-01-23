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

from pytket import Circuit, Qubit, Bit, OpType
from pytket.backends import Backend
from pytket.backends.backendresult import BackendResult
from pytket.utils.outcomearray import OutcomeArray  # type: ignore
from copy import copy
from typing import List, Dict, Tuple, Counter, cast, Sequence
from qermit import MitTask, CircuitShots, TaskGraph, MitRes  # type: ignore


def get_leakage_gadget_circuit(
    circuit_qubit: Qubit, postselection_qubit: Qubit, postselection_bit: Bit
) -> Circuit:
    c = Circuit()
    c.add_qubit(circuit_qubit)
    c.add_qubit(postselection_qubit)
    c.add_bit(postselection_bit)
    c.X(postselection_qubit)
    c.add_barrier([circuit_qubit, postselection_qubit])
    c.H(postselection_qubit).ZZMax(postselection_qubit, circuit_qubit)
    c.add_barrier([circuit_qubit, postselection_qubit])
    c.ZZMax(postselection_qubit, circuit_qubit).H(postselection_qubit).Z(circuit_qubit)
    c.add_barrier([circuit_qubit, postselection_qubit])
    c.Measure(postselection_qubit, postselection_bit)
    c.add_gate(OpType.Reset, [postselection_qubit])
    return c


def get_detection_circuit(
    original_circuit: Circuit, max_circuit_qubits: int
) -> Tuple[Circuit, List[Tuple[Bit, ...]]]:
    n_qubits = original_circuit.n_qubits
    if n_qubits == 0:
        raise ValueError(
            "Circuit for Leakage Gadget Postselection must have at least one Qubit."
        )
    # In generating the scheme from a Backend, sometimes a simulator Backend
    # may be passed through which has "no qubits", but in practice can have "infinite"
    # Assuming no real device would have maximum 0 qubits, if max_circuit_qubits
    # is set to 0, we assume its a simulator and so set it to 2*n_qubits, a value
    # that means each qubit can have an ancilla for completing leakage detection
    # with
    max_circuit_qubits = 2 * n_qubits if max_circuit_qubits == 0 else max_circuit_qubits
    n_spare_qubits = max_circuit_qubits - n_qubits
    if n_spare_qubits <= 0:
        raise ValueError(
            "Circuit has more or equal Qubits to the parameter maximum allowed."
        )
    # construct a new circuit by building up a new Circuit
    detection_circuit = Circuit()
    postselection_qubits: List[Qubit] = [
        Qubit("leakage_gadget_qubit", i) for i in range(n_spare_qubits)
    ]
    for q in original_circuit.qubits + postselection_qubits:
        detection_circuit.add_qubit(q)
    for b in original_circuit.bits:
        detection_circuit.add_bit(b)

    # construct a Circuit that is the original Circuit without end of Circuit Measure gates
    end_circuit_measures: Dict[Qubit, Bit] = {}
    for com in original_circuit.get_commands():
        if com.op.type == OpType.Barrier:
            detection_circuit.add_barrier(com.args)
            continue
        # first check if a mid circuit measure needs to be readded
        for q in com.qubits:
            # this condition only true if this Qubit has previously had a measure operation
            # this implies it is in another Quantum operation and thus previous measure was "mid-circuit"
            if q in end_circuit_measures:
                detection_circuit.Measure(q, end_circuit_measures.pop(q))
        if com.op.type == OpType.Measure:
            # can assume it only has one Qubit and one Bit as is a Measure op
            # if mid measure then will be rewritten
            end_circuit_measures[com.qubits[0]] = com.bits[0]
        elif com.op.params:
            detection_circuit.add_gate(com.op.type, com.op.params, com.args)
        else:
            detection_circuit.add_gate(com.op.type, com.args)

    # for each entry in end_circuit_measures, we want to add a leakage_gadget_circuit
    # we try to use each free architecture qubit as few times as possible
    q_ps_index = 0
    b_ps_index = 0
    postselection_bits: List[Bit] = []
    for q in end_circuit_measures:
        if q.reg_name == "leakage_gadget_qubit":
            raise ValueError(
                "Leakage Gadget scheme makes a qubit register named 'leakage_gadget_qubit' but this already exists in the passed circuit."
            )
        if q_ps_index == n_spare_qubits:
            q_ps_index = 0
        leakage_gadget_bit = Bit("leakage_gadget_bit", b_ps_index)
        leakage_gadget_circuit = get_leakage_gadget_circuit(
            q, postselection_qubits[q_ps_index], leakage_gadget_bit
        )
        detection_circuit.append(leakage_gadget_circuit)
        postselection_bits.append(leakage_gadget_bit)
        # increment values for adding postselection to
        b_ps_index += 1
        q_ps_index += 1

    # finally we measure the original qubits
    for q, b in end_circuit_measures.items():
        detection_circuit.Measure(q, b)

    detection_circuit.remove_blank_wires()
    return detection_circuit, postselection_bits


def postselection_circuits_task_gen(max_circuit_qubits: int) -> MitTask:
    """

    Returns a MitTask object that produces post selection circuits for
    a leakage detection gadget.

    :param max_circuit_qubits: Total number of qubits available on Backend Circuit being run through
    :type max_circuit_qubits: int
    """

    def task(
        obj, circs_shots: List[CircuitShots]
    ) -> Tuple[List[CircuitShots], List[Bit]]:
        """
        :param circ_shots: A list of tuple of circuits and shots. Each circuit has postselection applied
        :type circ_shots: List[CircuitShots]

        :return: Postselection circuits
        :rtype: Tuple[List[CircuitShots]]
        """
        all_postselection_circs_shots = []
        all_postselection_bits = []
        for circ, shots in circs_shots:
            new_circuit, postselection_bits = get_detection_circuit(
                circ, max_circuit_qubits
            )
            all_postselection_circs_shots.append(CircuitShots(new_circuit, shots))
            all_postselection_bits.append(postselection_bits)
        return (
            all_postselection_circs_shots,
            all_postselection_bits,
        )

    return MitTask(
        _label="GeneratePostselectionCircuits",
        _n_in_wires=1,
        _n_out_wires=2,
        _method=task,
    )


def postselection_results_task_gen() -> MitTask:
    """
    Returns a MitTask object that postselects on output results.

    :return: MitTask object discard results detecting leakage.
    :rtype: MitTask
    """

    def task(
        obj,
        all_results: List[BackendResult],
        all_postselection_bits: List[List[Bit]],
    ) -> Tuple[List[BackendResult]]:
        """
        :param all_results: Results being postselected on
        :type all_results: List[BackendResult]

        :return: Postselected results
        :rtype: Tuple[List[BackendResult]]
        """
        postselected_results: List[BackendResult] = []
        for result, postselection_bits in zip(all_results, all_postselection_bits):
            # get counts
            received_counts: Counter[Tuple[int, ...]] = result.get_counts()
            # make empty counter object for adding amended results to
            new_counts: Counter[Tuple[int, ...]] = Counter()

            #
            postselection_indices: List[int] = [
                result.c_bits[b] for b in postselection_bits
            ]
            # as we delete key as we go, go through state from back to front
            # to avoid
            postselection_indices.sort(reverse=True)
            for state in received_counts:
                # first of all find the condensed state without ancilla bits
                experiment_key = list(state)
                banned_state = False
                for i in postselection_indices:
                    if experiment_key[i] == 1:
                        banned_state = True
                        break

                    del experiment_key[i]

                if not banned_state:
                    new_counts[tuple(experiment_key)] = received_counts[state]

            # find remaining bits
            remaining_bits: List[Bit] = list(result.c_bits.keys())
            for i in postselection_indices:
                del remaining_bits[i]
            # make counter object
            outcome_array = {
                OutcomeArray.from_readouts([key]): val
                for key, val in new_counts.items()
            }

            outcome_counts = Counter(outcome_array)
            # add to results
            postselected_results.append(
                BackendResult(
                    counts=outcome_counts,
                    c_bits=cast(Sequence[Bit], cast(Sequence[Bit], remaining_bits)),
                )
            )
        return (postselected_results,)

    return MitTask(
        _label="PostselectResults",
        _n_in_wires=2,
        _n_out_wires=1,
        _method=task,
    )


def gen_Leakage_Detection_MitRes(backend: Backend, **kwargs) -> MitRes:
    """
    Produces a MitRes object that applies Leakagae Gadget Detection Postselection techniques to experiment circuits.

    :param backend: Backend which experiments are default executed through.
    :type backend: Backend
    :key mitres: MitRes object postselection MitRes is built around if given.
    """
    _mitres = copy(
        kwargs.get("mitres", MitRes(backend, _label="LeakageDetectionMitRes"))
    )

    _task_graph_mitres = TaskGraph().from_TaskGraph(_mitres)
    _task_graph_mitres.add_wire()
    _task_graph_mitres.prepend(
        postselection_circuits_task_gen(len(backend.backend_info.architecture.nodes))  # type: ignore
    )
    _task_graph_mitres.append(postselection_results_task_gen())
    for n in _mitres._task_graph.nodes:
        if hasattr(n, "_label"):
            n._label = "PS" + n._label
    return MitRes(backend).from_TaskGraph(_task_graph_mitres)

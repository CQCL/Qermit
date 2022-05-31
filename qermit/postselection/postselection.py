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
from copy import copy
from pytket.utils.outcomearray import OutcomeArray  # type: ignore
from typing import List, Tuple, Counter, Dict, Set, cast, Sequence
from itertools import combinations

from sympy import intersection
from qermit import MitTask, MitRes, CircuitShots


def transform_circuit(
    base_circuit: Circuit,
    postselection_circuit: Circuit,
    logical_qubits: List[Qubit],
    ancillas: List[Tuple[Qubit, Bit]],
) -> Tuple[Circuit, List[Tuple[Bit, ...]]]:
    # make a complete new circuit for adding gates to
    new_circuit = Circuit()
    for q in base_circuit.qubits:
        new_circuit.add_qubit(q)
    postselection_bits = postselection_circuit.bits
    for b in base_circuit.bits:
        # TODO: add handling as instead of raising error
        if b in postselection_bits:
            raise ValueError(
                "Bit in Postselection circuit also in original experimental circuit."
            )
        new_circuit.add_bit(b)

    end_circuit_measures: Dict[Qubit, Bit] = {}
    for com in base_circuit.get_commands():
        if com.op.type == OpType.Measure:
            # can assume it only has one Qubit and one Bit as a Measure op
            # if mid measure then will be rewritten
            end_circuit_measures[com.qubits[0]] = com.bits[0]
        # just add command to circuit TODO: surely a better way than this..
        elif com.op.params:
            new_circuit.add_gate(com.op.type, com.op.params, com.args)
        else:
            new_circuit.add_gate(com.op.type, com.args)

    # assumes that all end of circuit measures in original circuit will have postselection, if postselection
    # acts over more than one logical qubit then do for all permutations
    # TODO: n! / (n - k) ! scaling, this will get nasty, have error handling for this.
    counter = 0
    new_postselection_bits: List[Tuple[Bit, ...]] = []
    for comb in combinations(list(end_circuit_measures.keys()), len(logical_qubits)):
        postselection_circuit_copy = postselection_circuit.copy()
        postselection_circuit_copy.rename_units(
            {a[0]: a[1] for a in zip(logical_qubits, comb)}
        )
        relabelled_bits_dict = {b: Bit(b.index, b.reg_name + str(counter)) for b in postselection_bits}
        postselection_circuit_copy.rename_units(relabelled_bits_dict)
        new_postselection_bits.append(list(relabelled_bits_dict.values()))
        new_circuit.append(postselection_circuit_copy)
        counter += 1
    # add back end of circuit measures
    for q in end_circuit_measures:
        new_circuit.Measure(q, b)
    return (new_circuit, new_postselection_bits)


def postselection_circuits_task_gen(
    postselection_circuit: Circuit,
    logical_qubits: List[Qubit],
    ancillas: List[Tuple[Qubit, Bit]],
) -> MitTask:
    """

    Returns a MitTask object that produces post selection circuits for
    some postselection gadget.

    :param postselection_circuit: Postselection gadet as Circuit
    :type postselection_circuit: Circuit
    :param logical_qubits: Qubits in postselection circuit which correspond to logical qubits in
        circuit
    :type logical_qubits: List[Qubit]
    :param ancillas: Bit in circuit that are ancillas
    :type ancillas: List[Bit]
    """
    # # if postselection circuit measures a logical qubit then invalid
    # if (
    #     len(intersection(postselection_circuit.qubit_to_bit_map.keys(), logical_qubits))
    #     > 0
    # ):
    #     raise ValueError("Given postselection circuit has measures on logical qubit.")

    def task(obj, circs_shots: List[CircuitShots]) -> Tuple[List[CircuitShots]]:
        """
        :param circ_shots: A list of tuple of circuits and shots. Each circuit has postselection applied
        :type circ_shots: List[CircuitShots]

        :return: Postselection circuits
        :rtype: Tuple[List[CircuitShots]]
        """
        all_postselection_circs_shots = []
        for circ, shots in circs_shots:
            new_circuit = transform_circuit(
                circ, postselection_circuit, logical_qubits, ancillas
            )
            all_postselection_circs_shots.append(CircuitShots(new_circuit, shots))
        return (all_postselection_circs_shots,)

    return MitTask(
        _label="GeneratePostselectionCircuits",
        _n_in_wires=1,
        _n_out_wires=1,
        _method=task,
    )


def postselection_results_task_gen(banned_results: Set[Tuple[bool, ...]]
) -> MitTask:
    """
    Returns a MitTask object that postselects on output results given some set
    of Bit with some banned results.

    :param postselection_bits: Bits being postselected over
    :type postselection_bits: List[Bit]
    :param banned_results: Which boolean strings Bits can not have counts for
    :type banned_results: Set[Tuple[bool, ...]]

    :return: MitTask object completing defined postselection
    :rtype: MitTask
    """

    def task(
        obj,
        all_results: List[BackendResult],
        all_postselection_bits: List[List[Tuple[Bit, ...]]]
    ) -> Tuple[List[BackendResult]]:
        """
        :param all_results: Results being postselected on
        :type all_results: List[BackendResult]

        :return: Postselected results
        :rtype: Tuple[List[BackendResult]]
        """
        postselected_results: List[BackendResult] = []
        for r, postselection_bits in zip(all_results, all_postselection_bits):

            received_counts: Counter[Tuple[int, ...]] = r.get_counts()
            new_counts: Counter[Tuple[int, ...]] = Counter()

            # for sub_bits in postselection_bits:
            # need to amend counts key tuples
            # for sub_bits in postselection_bits:
            #     for b in 
                # postselection_indices.append([r.c_bits[b] for b in sub_bits])
            # postselection_indices = [r.c_bits[b] for sub_bits in postselection_bits for b in sub_bits]
            # postselection_indices.sort(reverse=True)


            print(r.get_counts())
            all_postselection_indices: List[int] = []
            for k in received_counts:
                banned = False
                for sub_bits in postselection_bits:
                    postselection_indices = [r.c_bits[b] for b in sub_bits]
                    postselection_key = tuple([k[i] for i in postselection_indices])
                    all_postselection_indices.extend(postselection_indices)
                    if postselection_key in banned_results:
                        banned = True
                        break
                experiment_key = list(k)
                # definitely a better way of doing this...
                for i in postselection_indices:  # note this is descending
                    del experiment_key[i]
                if postselection_key in banned_results:
                    new_counts[tuple(experiment_key)] = 0
                else:
                    new_counts[tuple(experiment_key)] = received_counts[k]

            # also need to work out remaining bits
            remaining_bits: List[Bit] = list(r.c_bits.keys())
            for i in postselection_indices:
                del remaining_bits[i]


            # also need to remove ancilla bit results
            outcome_array = {
                OutcomeArray.from_readouts([key]): val
                for key, val in new_counts.items()
            }
            outcome_counts = Counter(outcome_array)
            # print(outcome_counts)
            print(remaining_bits)
            print(new_counts)
            postselected_results.append(
                BackendResult(
                    counts=outcome_counts,
                    c_bits=cast(Sequence[Bit], cast(Sequence[Bit], remaining_bits)),
                )
            )
        return (postselected_results,)

    return MitTask(
        _label="PostselectResults",
        _n_in_wires=1,
        _n_out_wires=1,
        _method=task,
    )


def gen_Postselection_MitRes(
    backend: Backend,
    postselection_circuit: Circuit,
    postselection_ancillas: List[Tuple[Qubit, Bit]],
    postselection_logical_qubits: List[Qubit],
    banned_results: Tuple[Tuple[Bit, ...], Set[Tuple[bool, ...]]],
    **kwargs
) -> MitRes:
    """
    Produces a MitRes object that applies Postselection techniques to experiment circuits. Each n-qubit Postselection circuit is attached to each n-qubit combination of all qubits in the experimental with end of circuit measures. Note that for k logical qubits in the experimental circuit there are n!/(n-k)! combinations to apply.

    :param backend: Backend which experiments are default executed through.
    :type backend: Backend
    :param postselection_circuit: Circuit gadget defining some postselection scheme
    :type postselection_circuit: Circuit
    :param postselection_ancillas: Which Qubit in postselection_circuit are ancillas, which Bit they measure to
    :type postselection_ancillas: List[Tuple[Qubit, Bit]]
    :param postselection_logical_qubits: Which Qubit in postselection_circuit should be attributed to logical qubits in experient circuits.
    :type postselection_logical_qubits: List[Qubit]
    :param banned_results: Which returned shots over which bits are not allowed and should be removed.
    :type banned_results: Tuple[Tuple[Bit, ...], List[Tuple[bool, ...]]]

    :key mitres: MitRes object postselection MitRes is built around if given.
    """

    _mitres = copy(kwargs.get("mitres", MitRes(backend, _label="PostselectionMitRes")))
    _mitres.prepend(
        postselection_circuits_task_gen(
            postselection_circuit, postselection_logical_qubits, postselection_ancillas
        )
    )
    _mitres.append(postselection_results_task_gen(banned_results[0], banned_results[1]))
    for n in _mitres._task_graph.nodes:
        if hasattr(n, "_label"):
            n._label = "PS" + n._label
    return _mitres

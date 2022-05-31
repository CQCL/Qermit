# Copyright 2019-2021 Cambridge Quantum Computing
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


import networkx as nx  # type: ignore
from qermit.spam.partial_spam_correction import (  # type: ignore
    gen_state_polarisation_dicts,
    gen_partial_tomography_circuits,
    partial_spam_setup_task_gen,
    partial_correlated_spam_circuits_task_gen,
    characterise_correlated_spam_task_gen,
    correct_partial_correlated_spam_task_gen,
)
from pytket.circuit import Qubit, CircBox, Circuit, Node  # type: ignore
from pytket.routing import Architecture  # type: ignore
from pytket.extensions.qiskit import AerBackend  # type: ignore
from numpy import identity


def test_gen_state_polarisation_dicts():
    g0 = nx.Graph()
    qbs = [Qubit(i) for i in range(5)]
    cons0 = [(qbs[0], qbs[1]), (qbs[2], qbs[1]), (qbs[2], qbs[3]), (qbs[3], qbs[4])]
    g0.add_edges_from(cons0)
    dicts0 = gen_state_polarisation_dicts(g0)
    assert len(dicts0) == 1
    assert dicts0[0][qbs[0]] != dicts0[0][qbs[1]]
    assert dicts0[0][qbs[1]] != dicts0[0][qbs[2]]
    assert dicts0[0][qbs[2]] != dicts0[0][qbs[3]]
    assert dicts0[0][qbs[3]] != dicts0[0][qbs[4]]

    g1 = nx.Graph()
    cons1 = [
        (qbs[0], qbs[1]),
        (qbs[2], qbs[1]),
        (qbs[0], qbs[2]),
        (qbs[2], qbs[3]),
        (qbs[3], qbs[4]),
        (qbs[2], qbs[4]),
    ]
    g1.add_edges_from(cons1)
    dicts1 = gen_state_polarisation_dicts(g1)
    assert len(dicts1) == 2

    assert dicts1[0][qbs[1]] != dicts1[0][qbs[2]]
    assert dicts1[0][qbs[3]] != dicts1[0][qbs[2]]
    assert dicts1[0][qbs[3]] != dicts1[0][qbs[4]]
    assert dicts1[1][qbs[0]] != dicts1[1][qbs[1]]
    assert dicts1[1][qbs[0]] != dicts1[1][qbs[2]]
    assert dicts1[1][qbs[4]] != dicts1[1][qbs[2]]


def test_gen_partial_tomography_circuits():
    cb = CircBox(Circuit(1).X(0))
    g0 = nx.Graph()
    qbs = [Qubit(i) for i in range(5)]
    cons0 = [(qbs[0], qbs[1]), (qbs[2], qbs[1]), (qbs[2], qbs[3]), (qbs[3], qbs[4])]
    g0.add_edges_from(cons0)
    res0 = gen_partial_tomography_circuits(gen_state_polarisation_dicts(g0), cb)
    assert len(res0[0]) == 4
    assert len(res0[0][0].get_commands()) == 6
    assert len(res0[0][1].get_commands()) == 8
    assert len(res0[0][2].get_commands()) == 9
    assert len(res0[0][3].get_commands()) == 11

    for x in res0[1]:
        assert len(res0[1][x]) == 4

    g1 = nx.Graph()
    cons1 = [
        (qbs[0], qbs[1]),
        (qbs[2], qbs[1]),
        (qbs[0], qbs[2]),
        (qbs[2], qbs[3]),
        (qbs[3], qbs[4]),
        (qbs[2], qbs[4]),
    ]
    g1.add_edges_from(cons1)
    res1 = gen_partial_tomography_circuits(gen_state_polarisation_dicts(g1), cb)
    assert len(res1[0]) == 8


def test_partial_spam_setup_task_gen():
    b = AerBackend()
    b._characterisation = dict()
    task = partial_spam_setup_task_gen(b, 1)
    assert task.n_in_wires == 1
    assert task.n_out_wires == 3
    c = Circuit(2).CX(0, 1).measure_all()
    wire = [(c, 10)]
    res = task([wire])
    assert len(res) == task.n_out_wires
    assert res[0] == [(c, 10)]
    assert res[1] == [c.qubit_to_bit_map]
    assert res[2] == True


def test_partial_correlated_spam_circuits_task_gen():
    b = AerBackend()
    qbs = [Node(i) for i in range(5)]
    cons = [
        (qbs[0], qbs[1]),
        (qbs[2], qbs[1]),
        (qbs[0], qbs[2]),
        (qbs[2], qbs[3]),
        (qbs[3], qbs[4]),
        (qbs[2], qbs[4]),
    ]
    b.backend_info.architecture = Architecture(cons)
    task = partial_correlated_spam_circuits_task_gen(b, 10, 1)
    assert task.n_in_wires == 1
    assert task.n_out_wires == 2
    false_res = task([False])
    assert len(false_res) == 2
    assert false_res[0] == []
    assert false_res[1] == dict()

    true_res = task([True])
    assert len(true_res) == 2
    assert len(true_res[0]) == 8
    assert len(true_res[1]) == 9


def test_characterisation_correction():
    b = AerBackend()
    qbs = [Node(i) for i in range(5)]
    cons = [
        (qbs[0], qbs[1]),
        (qbs[2], qbs[1]),
        (qbs[0], qbs[2]),
        (qbs[2], qbs[3]),
        (qbs[3], qbs[4]),
        (qbs[2], qbs[4]),
    ]
    b.backend_info.architecture = Architecture(cons)
    res = partial_correlated_spam_circuits_task_gen(b, 10, 1)([True])
    circs = [r[0] for r in res[0]]
    handles = b.process_circuits(circs, 10)
    results = b.get_results(handles)

    char_task = characterise_correlated_spam_task_gen(b, 1, 10)
    assert char_task.n_in_wires == 2
    assert char_task.n_out_wires == 1

    wire = (results, res[1])
    char_res = char_task(wire)
    assert "CorrelatedSpamCorrection" in b.backend_info.misc
    assert char_res == (True,)
    char = b.backend_info.misc["CorrelatedSpamCorrection"]
    assert char.Distance == 1
    identity_matrix = identity(4)
    for x in char.CorrelationToMatrix:
        assert char.CorrelationToMatrix[x].all() == identity_matrix.all()
    assert len(char.CorrelatedEdges) == 9

    cor_task = correct_partial_correlated_spam_task_gen(b)
    assert cor_task.n_in_wires == 3
    assert cor_task.n_out_wires == 1

    c0 = Circuit()
    c0.add_qubit(qbs[0])
    c0.add_qubit(qbs[1])
    c0.add_qubit(qbs[3])
    c0.add_qubit(qbs[4])
    c1 = Circuit()
    c1.add_qubit(qbs[2])
    c1.add_qubit(qbs[0])
    c1.add_qubit(qbs[4])
    c0.X(qbs[0]).X(qbs[3]).measure_all()
    c1.X(qbs[2]).X(qbs[4]).measure_all()

    ex_handles = b.process_circuits([c0, c1], 10)
    ex_res = b.get_results(ex_handles)
    cor_res = cor_task([ex_res, [c0.qubit_to_bit_map, c1.qubit_to_bit_map], True])
    assert cor_res[0][0].get_counts()[(1, 0, 1, 0)] == 10
    assert cor_res[0][1].get_counts()[(0, 1, 1)] == 10


if __name__ == "__main__":
    test_gen_state_polarisation_dicts()
    test_gen_partial_tomography_circuits()
    test_partial_spam_setup_task_gen()
    test_partial_correlated_spam_circuits_task_gen()
    test_characterisation_correction()

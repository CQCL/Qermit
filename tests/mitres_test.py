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


from qermit import (  # type: ignore
    MitRes,
    CircuitShots,
)
from qermit.taskgraph.mitres import (  # type: ignore
    backend_handle_task_gen,
    backend_res_task_gen,
    split_shots_task_gen,
    group_shots_task_gen,
    gen_shot_split_MitRes,
)
from pytket import Circuit
from pytket.extensions.qiskit import AerBackend  # type: ignore


def test_backend_handle_result_task_gen():
    b = AerBackend()
    handle_task = backend_handle_task_gen(b)
    assert handle_task.n_in_wires == 1
    assert handle_task.n_out_wires == 1

    c0 = Circuit(2).CX(0, 1).measure_all()
    c1 = Circuit(2).X(0).X(1).measure_all()
    test_handles = handle_task([[(c0, 10), (c1, 20)]])[0]
    assert len(test_handles) == 2

    results_task = backend_res_task_gen(b)
    assert results_task.n_in_wires == 1
    assert results_task.n_out_wires == 1

    test_results = results_task([test_handles])[0]
    assert len(test_results) == 2
    assert test_results[0].get_counts()[(0, 0)] == 10
    assert test_results[1].get_counts()[(1, 1)] == 20


def test_mitres_run():
    b = AerBackend()
    mr = MitRes(b, _label="TestLabel")
    assert str(mr) == "<MitRes::TestLabel>"
    c0 = Circuit(2).CX(0, 1).measure_all()
    c1 = Circuit(2).X(0).X(1).measure_all()

    res = mr.run([(c0, 10), (c1, 20)])
    assert len(res) == 2
    assert res[0].get_counts()[(0, 0)] == 10
    assert res[1].get_counts()[(1, 1)] == 20


def test_split_shots_task_gen():

    n_shots_1 = 20
    circ_1 = Circuit(1).X(0)
    circ_shots_1 = CircuitShots(circ_1, n_shots_1)

    n_shots_2 = 31
    circ_2 = Circuit(2).CX(0, 1)
    circ_shots_2 = CircuitShots(circ_2, n_shots_2)

    max_shots = 10
    task = split_shots_task_gen(max_shots)

    assert task.n_in_wires == 1
    assert task.n_out_wires == 2

    test_result = task([[circ_shots_1, circ_shots_2]])

    assert test_result[1] == [0, 0, 1, 1, 1, 1]

    circ_1_test_result = [
        circ_shots
        for i, circ_shots in enumerate(test_result[0])
        if test_result[1][i] == 0
    ]
    shot_total = 0
    for circ_shots in circ_1_test_result:
        assert circ_shots.Circuit == circ_1
        shot_total += circ_shots.Shots
    assert shot_total == n_shots_1

    circ_2_test_result = [
        circ_shots
        for i, circ_shots in enumerate(test_result[0])
        if test_result[1][i] == 1
    ]
    shot_total = 0
    for circ_shots in circ_2_test_result:
        assert circ_shots.Circuit == circ_2
        shot_total += circ_shots.Shots
    assert shot_total == n_shots_2


def test_group_shots_task_gen():

    backend = AerBackend()

    circ_1 = Circuit(1).X(0).measure_all()
    circ_2 = Circuit(2).CX(0, 1).measure_all()
    circ_3 = Circuit(3).CX(0, 1).CZ(1, 2).H(1).measure_all()

    circ_1_results_1 = backend.run_circuit(circ_1, 5)
    circ_1_results_2 = backend.run_circuit(circ_1, 3)

    circ_2_results_1 = backend.run_circuit(circ_2, 4)

    circ_3_results_1 = backend.run_circuit(circ_3, 2)
    circ_3_results_2 = backend.run_circuit(circ_3, 6)
    circ_3_results_3 = backend.run_circuit(circ_3, 3)

    task = group_shots_task_gen()
    merged_results = task(
        (
            [
                circ_1_results_1,
                circ_1_results_2,
                circ_2_results_1,
                circ_3_results_1,
                circ_3_results_2,
                circ_3_results_3,
            ],
            [0, 0, 1, 2, 2, 2],
        )
    )

    assert len(merged_results[0][0].get_shots()) == 8
    assert len(merged_results[0][1].get_shots()) == 4
    assert len(merged_results[0][2].get_shots()) == 11


def test_gen_shot_split_MitRes():

    backend = AerBackend()

    mitres = gen_shot_split_MitRes(backend, 5)
    mitres.get_task_graph()

    n_shots_1 = 8
    circ_1 = Circuit(1).X(0).measure_all()
    n_shots_2 = 12
    circ_2 = Circuit(2).CX(0, 1).measure_all()

    results = mitres.run(
        [CircuitShots(circ_1, n_shots_1), CircuitShots(circ_2, n_shots_2)]
    )

    assert len(results[0].get_shots()) == n_shots_1
    assert len(results[1].get_shots()) == n_shots_2


if __name__ == "__main__":
    test_backend_handle_result_task_gen()
    test_mitres_run()
    test_split_shots_task_gen()
    test_group_shots_task_gen()
    test_gen_shot_split_MitRes()

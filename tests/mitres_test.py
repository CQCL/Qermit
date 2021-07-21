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
)
from qermit.taskgraph.mitres import (  # type: ignore
    backend_handle_task_gen,
    backend_res_task_gen,
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


if __name__ == "__main__":
    test_backend_handle_result_task_gen()
    test_mitres_run()

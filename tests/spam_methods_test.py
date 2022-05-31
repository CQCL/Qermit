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


from pytket import Circuit, Qubit
from pytket.extensions.qiskit import AerBackend  # type: ignore
from pytket.circuit import Node  # type: ignore
from pytket.architecture import Architecture  # type: ignore
from qermit.spam import (  # type: ignore
    gen_FullyCorrelated_SPAM_MitRes,
    gen_UnCorrelated_SPAM_MitRes,
    CorrectionMethod,
)
from qermit.taskgraph import CircuitShots  # type: ignore


def gen_test_wire():
    c0 = Circuit(4).X(0).X(2).measure_all()
    c1 = Circuit(5).X(1).X(3).measure_all()
    return [CircuitShots(c0.copy(), 20), CircuitShots(c1.copy(), 20)]


def test_gen_FC_mr():
    experiment_wire = gen_test_wire()
    b = AerBackend()
    qb_subsets = [
        [Qubit(0), Qubit(1)],
        [Qubit(2), Qubit(3)],
        [Qubit(4)],
        [Qubit(5), Qubit(6), Qubit(7)],
    ]

    mr = gen_FullyCorrelated_SPAM_MitRes(
        b, 100, qb_subsets, corr_method=CorrectionMethod.Invert
    )
    res = mr.run(experiment_wire)
    assert len(res) == 2
    assert res[0].get_counts()[(1, 0, 1, 0)] == 20
    assert res[1].get_counts()[(0, 1, 0, 1, 0)] == 20


def test_gen_UC_mr():
    experiment_wire = gen_test_wire()
    b = AerBackend()
    conn = [
        (Node("q", 0), Node("q", 1)),
        (Node("q", 2), Node("q", 1)),
        (Node("q", 0), Node("q", 3)),
        (Node("q", 3), Node("q", 4)),
        (Node("q", 4), Node("q", 5)),
        (Node("q", 2), Node("q", 5)),
        (Node("q", 1), Node("q", 4)),
    ]
    b.backend_info.architecture = Architecture(conn)
    mr = gen_UnCorrelated_SPAM_MitRes(b, 100)
    res = mr.run(experiment_wire)
    assert len(res) == 2
    assert res[0].get_counts()[(1, 0, 1, 0)] == 20
    assert res[1].get_counts()[(0, 1, 0, 1, 0)] == 20


if __name__ == "__main__":
    test_gen_FC_mr()
    test_gen_UC_mr()

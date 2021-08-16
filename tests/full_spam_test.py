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


from pytket import Circuit, Qubit, Bit
from qermit.spam.full_spam_correction import (  # type: ignore
    gen_full_tomography_spam_circuits_task,
    gen_full_tomography_spam_characterisation_task,
    gen_full_tomography_spam_correction_task,
    gen_get_bit_maps_task,
)
from qermit.spam import (  # type: ignore
    CorrectionMethod,
)
from pytket.extensions.qiskit import AerBackend  # type: ignore
import numpy as np


def test_gen_full_tomography_spam_circuits_task():
    b = AerBackend()
    b._characterisation = dict()
    task = gen_full_tomography_spam_circuits_task(
        b, 5, [[Qubit(0), Qubit(1)], [Qubit(2), Qubit(3)]]
    )
    assert task.n_in_wires == 1
    assert task.n_out_wires == 3

    c0 = Circuit(3).CX(0, 1).X(2).measure_all()
    c1 = Circuit(2).X(0).X(1).measure_all()
    wire = [(c0, 10), (c1, 20)]
    res = task([wire])
    assert len(res) == 3
    assert res[0] == wire
    assert len(res[1]) == len(res[2])
    assert len(res[1][0][0].get_commands()) == 6
    assert len(res[1][1][0].get_commands()) == 8
    assert len(res[1][2][0].get_commands()) == 8
    assert len(res[1][3][0].get_commands()) == 10


def test_full_tomography_spam_characterisation_task_gen():
    b = AerBackend()
    b._characterisation = dict()
    qb_subsets = [[Qubit(0), Qubit(1)], [Qubit(2), Qubit(3)]]

    c0 = Circuit(3).CX(0, 1).X(2).measure_all()
    c1 = Circuit(2).X(0).X(1).measure_all()
    wire = [(c0, 10), (c1, 20)]
    spam_info = gen_full_tomography_spam_circuits_task(b, 5, qb_subsets)([wire])

    spam_circs = [c[0] for c in spam_info[1]]

    handles = b.process_circuits(spam_circs, 5)
    results = b.get_results(handles)

    task = gen_full_tomography_spam_characterisation_task(b, qb_subsets)
    assert task.n_in_wires == 2
    assert task.n_out_wires == 1

    task_res = task([results, spam_info[2]])
    assert task_res == (True,)
    char = b.backend_info.misc["FullCorrelatedSpamCorrection"]
    assert char[0] == qb_subsets
    assert char[1] == {
        Qubit(0): (0, 0),
        Qubit(1): (0, 1),
        Qubit(2): (1, 0),
        Qubit(3): (1, 1),
    }
    assert char[2][0].all() == np.identity(2).all()
    assert char[2][1].all() == np.identity(2).all()


def test_gen_get_bit_maps_task():
    task = gen_get_bit_maps_task()
    assert task.n_in_wires == 1
    assert task.n_out_wires == 2

    c0 = Circuit(3).CX(0, 1).X(2).measure_all()
    c1 = Circuit(2, 2).X(0).Measure(0, 0).X(1).SWAP(0, 1).Measure(0, 1)
    wire = [(c0, 10), (c1, 50)]
    res = task([wire])
    assert len(res) == 2
    assert res[0] == wire
    comp0 = (c0.qubit_to_bit_map, {})
    comp1 = (c1.qubit_to_bit_map, {Bit(0): Qubit(0)})
    assert res[1][0] == comp0
    assert res[1][1] == comp1


def test_gen_full_tomography_spam_correction_task():
    # characterise noiseless matrix
    # use prior experiment
    b = AerBackend()
    b._characterisation = dict()
    qb_subsets = [[Qubit(0), Qubit(1)], [Qubit(2), Qubit(3)]]
    c0 = Circuit(3).CX(0, 1).X(2).measure_all()
    c1 = Circuit(2, 2).X(0).Measure(0, 0).X(1).SWAP(0, 1).Measure(0, 1)
    wire = [(c0, 10), (c1, 20)]
    spam_info = gen_full_tomography_spam_circuits_task(b, 5, qb_subsets)([wire])
    spam_circs = [c[0] for c in spam_info[1]]
    handles = b.process_circuits(spam_circs, 5)
    results = b.get_results(handles)
    # just returns bool
    gen_full_tomography_spam_characterisation_task(b, qb_subsets)(
        [results, spam_info[2]]
    )

    handles1 = b.process_circuits([c0, c1], 20)
    results1 = b.get_results(handles1)
    q_b_maps = [(c0.qubit_to_bit_map, {}), (c1.qubit_to_bit_map, {Bit(0): Qubit(0)})]
    task = gen_full_tomography_spam_correction_task(b, CorrectionMethod.Invert)
    assert task.n_in_wires == 3
    assert task.n_out_wires == 1

    wire = [results1, q_b_maps, True]
    corrected_results = task(wire)[0]
    assert len(corrected_results) == 2
    assert corrected_results[0].get_counts()[(0, 0, 1)] == 20
    assert corrected_results[1].get_counts()[(1, 1)] == 20


if __name__ == "__main__":
    test_gen_full_tomography_spam_circuits_task()
    test_full_tomography_spam_characterisation_task_gen()
    test_gen_full_tomography_spam_correction_task()
    test_gen_get_bit_maps_task()

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


from qermit.frame_randomisation import (  # type: ignore
    FrameRandomisation,
    gen_Frame_Randomisation_MitRes,
)
from qermit.frame_randomisation.frame_randomisation import (  # type: ignore
    frame_randomisation_circuits_task_gen,
)
from pytket import Circuit
from pytket.extensions.qiskit import AerBackend  # type: ignore


def test_frame_randomisation_circuits_task_gen():
    c = Circuit(2).CX(0, 1).Rx(0.289, 1).CX(0, 1).measure_all()
    ufr_task = frame_randomisation_circuits_task_gen(
        10, _fr_type=FrameRandomisation.UniversalFrameRandomisation
    )
    assert ufr_task.n_in_wires == 1
    assert ufr_task.n_out_wires == 1
    wire = [(c, 100), (c, 50)]
    ufr_res = ufr_task([wire])
    # make sure right number of samples of each circuit is produced
    assert len(ufr_res) == 1
    assert len(ufr_res[0]) == 20
    assert ufr_res[0][0][1] == 10
    assert ufr_res[0][10][1] == 5
    # should be same number of commands, + 2 barriers + 2 frame gates each side of the single cyle
    assert len(ufr_res[0][0][0].get_commands()) == len(c.get_commands()) + 6

    pfr_task = frame_randomisation_circuits_task_gen(
        10, _fr_type=FrameRandomisation.PauliFrameRandomisation
    )
    pfr_res = pfr_task([wire])
    assert len(pfr_res) == 1
    assert len(pfr_res[0]) == 20
    assert pfr_res[0][0][1] == 10
    assert pfr_res[0][10][1] == 5
    # should be same number of commands, + 2 barriers + 2 frame gates each side of the two cycles
    assert len(pfr_res[0][0][0].get_commands()) == len(c.get_commands()) + 12


def test_gen_Frame_Randomisation_MitRes():
    c = Circuit(2).X(0).CX(0, 1).measure_all()
    pfr_mitres = gen_Frame_Randomisation_MitRes(
        AerBackend(), 5, frame_randomisation=FrameRandomisation.PauliFrameRandomisation
    )
    assert pfr_mitres.n_in_wires == 1
    assert pfr_mitres.n_out_wires == 1
    wire = [(c, 100), (c, 50)]
    pfr_res = pfr_mitres.run(wire)
    assert len(pfr_res) == 2
    assert pfr_res[0].get_counts()[(1, 1)] == 100
    assert pfr_res[1].get_counts()[(1, 1)] == 50

    ufr_mitres = gen_Frame_Randomisation_MitRes(
        AerBackend(),
        5,
        frame_randomisation=FrameRandomisation.UniversalFrameRandomisation,
    )
    assert ufr_mitres.n_in_wires == 1
    assert ufr_mitres.n_out_wires == 1
    wire = [(c, 100), (c, 50)]
    ufr_res = ufr_mitres.run(wire)
    assert len(ufr_res) == 2
    assert ufr_res[0].get_counts()[(1, 1)] == 100
    assert ufr_res[1].get_counts()[(1, 1)] == 50


if __name__ == "__main__":
    test_frame_randomisation_circuits_task_gen()
    test_gen_Frame_Randomisation_MitRes()

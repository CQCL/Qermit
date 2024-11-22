# Copyright 2019-2023 Quantinuum
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
from pytket.extensions.quantinuum import QuantinuumBackend, QuantinuumAPIOffline
from qermit.frame_randomisation.h_series_randomisation import gen_h_series_randomised_circuit, get_wfh

from pytket.unit_id import BitRegister
from collections import Counter


def test_h_series_randomisation():
    # These tests check that the ideal behaviour of the circuits
    # is not altered by adding randomisation.

    api_offline = QuantinuumAPIOffline()
    backend = QuantinuumBackend(
        device_name="H1-1LE",
        api_handler = api_offline,
    )
    wasm_file_handler=get_wfh()

    # Small circuit with just one ZZMax.
    # The ideal output is 00.
    circuit = Circuit(3)
    meas_reg = BitRegister(name='measure', size=2)
    circuit.add_c_register(meas_reg)

    circuit.ZZMax(0,1)
    circuit.Measure(
        qubit=circuit.qubits[0],
        bit=meas_reg[0]
    )
    circuit.Measure(
        qubit=circuit.qubits[1],
        bit=meas_reg[1]
    )

    randomised_circuit = gen_h_series_randomised_circuit(circuit)
    compiled_circuit = backend.get_compiled_circuit(randomised_circuit, optimisation_level=0)

    n_shots = 100
    result = backend.run_circuit(
        compiled_circuit, 
        n_shots=n_shots,
        wasm_file_handler=wasm_file_handler,
        no_opt=True
    )
    assert result.get_counts(cbits=meas_reg) == Counter({(0,0,): n_shots})

    # To consecutive ZZMax gates.
    # Has the effect of acting Z_0 Z_1.
    # Checked by applying hadamard rotations.
    # I deal outcome in rotate basis is 11.
    circuit = Circuit(3)
    meas_reg = BitRegister(name='measure', size=2)
    circuit.add_c_register(meas_reg)

    circuit.H(0)
    circuit.H(1)
    
    circuit.ZZMax(0,1)
    circuit.ZZMax(0,1)
    
    circuit.H(0)
    circuit.H(1)
    
    circuit.Measure(
        qubit=circuit.qubits[0],
        bit=meas_reg[0]
    )
    circuit.Measure(
        qubit=circuit.qubits[1],
        bit=meas_reg[1]
    )

    randomised_circuit = gen_h_series_randomised_circuit(circuit)
    compiled_circuit = backend.get_compiled_circuit(randomised_circuit, optimisation_level=0)

    n_shots = 100
    result = backend.run_circuit(
        compiled_circuit, 
        n_shots=n_shots,
        wasm_file_handler=wasm_file_handler,
        no_opt=True
    )
    assert result.get_counts(cbits=meas_reg) == Counter({(1,1,): n_shots})

    # Slightly larger circuit.
    # Ideal outcome in rotated basis is 101.
    circuit = Circuit(3)
    meas_reg = BitRegister(name='measure', size=3)
    circuit.add_c_register(meas_reg)

    circuit.H(0)
    circuit.H(1)
    circuit.H(2)
    
    circuit.ZZMax(0,1)
    circuit.ZZMax(1,2)
    circuit.ZZMax(0,1)
    circuit.ZZMax(1,2)
    
    circuit.H(0)
    circuit.H(1)
    circuit.H(2)
    
    circuit.Measure(
        qubit=circuit.qubits[0],
        bit=meas_reg[0]
    )
    circuit.Measure(
        qubit=circuit.qubits[1],
        bit=meas_reg[1]
    )
    circuit.Measure(
        qubit=circuit.qubits[2],
        bit=meas_reg[2]
    )

    randomised_circuit = gen_h_series_randomised_circuit(circuit)
    compiled_circuit = backend.get_compiled_circuit(randomised_circuit, optimisation_level=0)

    n_shots = 100
    result = backend.run_circuit(
        compiled_circuit, 
        n_shots=n_shots,
        wasm_file_handler=wasm_file_handler,
        no_opt=True
    )
    assert result.get_counts(cbits=meas_reg) == Counter({(1,0,1,): n_shots})

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

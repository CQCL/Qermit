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


from typing import cast
import numpy as np
from qermit.clifford_noise_characterisation.cdr_post import (  # type: ignore
    _PolyCDRCorrect,
    cdr_calibration_task_gen,
    cdr_correction_task_gen,
    cdr_quality_check_task_gen,
)
from pytket.extensions.qiskit import AerBackend  # type: ignore
from pytket.pauli import QubitPauliString, Pauli  # type: ignore
from pytket.utils import QubitPauliOperator
from pytket import Qubit


def test_linear_cdr_calib() -> None:
    test_params = [1.2, 0.1]
    test_noisy_exp = cast(np.ndarray, np.linspace(0, 1, 30))

    exact_model = _PolyCDRCorrect(1, test_params)

    exact_correct = np.vectorize(exact_model.correct)
    test_exact_exp = exact_correct(test_noisy_exp)

    assert np.array_equal(
        test_exact_exp, test_params[0] * test_noisy_exp + test_params[1]
    )
    np.random.seed(12)
    test_exact_exp_error = test_exact_exp + np.random.normal(
        0, 0.01, len(test_noisy_exp)
    )

    test_model = _PolyCDRCorrect(1)

    test_model.calibrate(test_noisy_exp.tolist(), test_exact_exp_error)
    assert np.isclose(test_params, test_model.params, rtol=1e-1).all()
    new_noisy = np.linspace(0.1, 1.1, 20)
    new_corrected_vals = np.vectorize(test_model.correct)(new_noisy)
    assert np.isclose(new_corrected_vals, exact_correct(new_noisy), rtol=1e-1).all()


def test_cdr_quality_check_task_gen():

    qual_task = cdr_quality_check_task_gen(
        distance_tolerance=0.1, calibration_fraction=0.5
    )
    assert qual_task.n_in_wires == 2
    assert qual_task.n_out_wires == 2

    # set up dummy test wires
    qps_012 = QubitPauliString(
        [Qubit(0), Qubit(1), Qubit(2)], [Pauli.Z, Pauli.Z, Pauli.Z]
    )
    qps_01 = QubitPauliString([Qubit(0), Qubit(1)], [Pauli.Z, Pauli.Z])

    qpo_012_original_noisy = QubitPauliOperator({qps_012: 1.0})
    qpo_01_original_noisy = QubitPauliOperator({qps_01: 0.5})

    qpo_012_training_1_noisy = QubitPauliOperator({qps_012: 0.99})
    qpo_012_training_2_noisy = QubitPauliOperator({qps_012: 0.98})
    qpo_012_training_1_ideal = QubitPauliOperator({qps_012: 1})
    qpo_012_training_2_ideal = QubitPauliOperator({qps_012: 1})

    qpo_01_training_1_noisy = QubitPauliOperator({qps_01: 0.45})
    qpo_01_training_2_noisy = QubitPauliOperator({qps_01: 0.55})
    qpo_01_training_1_ideal = QubitPauliOperator({qps_01: 0.5})
    qpo_01_training_2_ideal = QubitPauliOperator({qps_01: 0.5})

    noisy_expectation = [qpo_012_original_noisy, qpo_01_original_noisy]
    state_circuit_exp = [
        [
            (qpo_012_training_1_noisy, qpo_012_training_1_ideal),
            (qpo_012_training_2_noisy, qpo_012_training_2_ideal),
        ],
        [
            (qpo_01_training_1_noisy, qpo_01_training_1_ideal),
            (qpo_01_training_2_noisy, qpo_01_training_2_ideal),
        ],
    ]

    noisy_expectation_return, state_circuit_exp_return = qual_task(
        [noisy_expectation, state_circuit_exp]
    )

    assert noisy_expectation_return == noisy_expectation
    assert state_circuit_exp == state_circuit_exp_return


def test_cdr_calibration_correction_task_gen():
    b = AerBackend()
    b._characterisation = dict()
    cal_task = cdr_calibration_task_gen(b, _PolyCDRCorrect(1))
    assert cal_task.n_in_wires == 1
    assert cal_task.n_out_wires == 1

    # set up dummy test wires
    qps_012 = QubitPauliString(
        [Qubit(0), Qubit(1), Qubit(2)], [Pauli.Z, Pauli.Z, Pauli.Z]
    )
    qps_01 = QubitPauliString([Qubit(0), Qubit(1)], [Pauli.Z, Pauli.Z])

    qpo_012 = QubitPauliOperator({qps_012: 1.0})
    qpo_01 = QubitPauliOperator({qps_01: 0.5})

    cal_wire = [
        [(qpo_012, qpo_012), (qpo_01, qpo_01)],
        [(qpo_01, qpo_01), (qpo_012, qpo_012)],
    ]
    # confirms calibration successful
    assert cal_task([cal_wire])
    assert "CDR_0" in b.backend_info.misc
    assert "CDR_1" in b.backend_info.misc

    model_0 = b.backend_info.misc["CDR_0"]
    model_1 = b.backend_info.misc["CDR_1"]
    # calibrated of 1.0, should always return same value
    assert np.isclose(model_0[qps_012].correct(1.0), 1.0, rtol=1e-1)
    assert np.isclose(model_0[qps_01].correct(1.0), 0.75, rtol=1e-1)
    assert np.isclose(model_1[qps_012].correct(1.0), 1.0, rtol=1e-1)
    assert np.isclose(model_1[qps_01].correct(1.0), 0.75, rtol=1e-1)

    cor_task = cdr_correction_task_gen(b)
    assert cor_task.n_in_wires == 2
    assert cor_task.n_out_wires == 1

    to_correct = [
        QubitPauliOperator({qps_01: 1.0, qps_012: 1.0}),
        QubitPauliOperator({qps_01: 1.0, qps_012: 1.0}),
    ]
    cor_res = cor_task((True, to_correct))[0]

    assert len(cor_res) == 2
    assert np.isclose(float(cor_res[0][qps_01]), 0.75, rtol=1e-1)
    assert np.isclose(float(cor_res[1][qps_01]), 0.75, rtol=1e-1)
    assert np.isclose(float(cor_res[0][qps_012]), 1.0, rtol=1e-1)
    assert np.isclose(float(cor_res[1][qps_012]), 1.0, rtol=1e-1)


if __name__ == "__main__":
    test_linear_cdr_calib()
    test_cdr_calibration_correction_task_gen()
    test_cdr_quality_check_task_gen()

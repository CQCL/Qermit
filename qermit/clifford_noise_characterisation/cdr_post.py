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


from abc import ABC, abstractmethod
from typing import List, Tuple, cast, Dict, Union

import numpy as np  #  type: ignore
import copy

from pytket.backends import Backend
from pytket.utils import QubitPauliOperator
from pytket.pauli import QubitPauliString  # type: ignore
from qermit import MitTask
import math
import warnings


class _BaseExCorrectModel(ABC):
    """
    Model for storing calibrations and correcting expectation values for some experiments.
    """

    @abstractmethod
    def calibrate(self, noisy_exp: List[float], exact_exp: List[float]) -> None:
        pass

    @abstractmethod
    def correct(self, noisy_expectation: float) -> float:
        return NotImplemented


class _PolyCDRCorrect(_BaseExCorrectModel):
    def __init__(self, degree: int, params: List[float] = []) -> None:
        super().__init__()
        if params:
            if len(params) != degree + 1:
                raise ValueError(
                    f"Degree {degree} polynomial requires {degree+1} parameters, "
                    f"{len(params)} were provided."
                )
        self.degree = degree
        self.params = params

    def calibrate(self, noisy_exp: List[float], exact_exp: List[float]) -> None:
        self.params = cast(
            List[float],
            cast(np.ndarray, np.polyfit(x=noisy_exp, y=exact_exp, deg=self.degree)).tolist(),
        )

    def correct(self, noisy_expectation: float) -> float:
        return cast(
            float,
            sum(
                coef * (noisy_expectation ** power)
                for coef, power in zip(self.params, range(self.degree, -1, -1))
            ),
        )

def cdr_quality_check_task_gen(distance_tolerance: float) -> MitTask:

    def cdr_quality_check_task(
        obj, 
        noisy_expectation: List[QubitPauliOperator], 
        state_circuit_exp: List[List[Tuple[QubitPauliOperator, QubitPauliOperator]]],
        ):

        print("cdr_quality_check_task noisy_expectation:", noisy_expectation)
        print("cdr_quality_check_task state_circuit_exp:", state_circuit_exp)

        for calibration, original in zip(state_circuit_exp, noisy_expectation):

            print("original dictionary:", original.to_list()[0]['coefficient'][0])

            original_coefficient = original.to_list()[0]['coefficient'][0]

            is_far_count = 0

            for qpo_pair in calibration:

                print("original", original, "qpo_pair", qpo_pair)

                noisy_qpo = qpo_pair[0]

                noisy_coefficient = noisy_qpo.to_list()[0]['coefficient'][0]

                if not math.isclose(noisy_coefficient, original_coefficient, abs_tol=distance_tolerance):
                    is_far_count += 1

            print("===== is_far_count", is_far_count)
            if is_far_count > len(calibration)/2:
                warnings.warn("Training data regularly differers significantly from original circuit. Fit may be poor.")

        return (noisy_expectation, state_circuit_exp, )

    return MitTask(
        _label="CDRQualityCheck",
        _n_in_wires=2,
        _n_out_wires=2,
        _method=cdr_quality_check_task,
    )


def cdr_calibration_task_gen(
    backend: Backend, model: _BaseExCorrectModel, tolerance: float
) -> MitTask:
    """
    Uses calibration results from running characterisation circuits through a device
    and a noiseless simulator to characterise some model for correcting noisy expectation values.

    :param backend: Backend for storing characterisation model in.
    :type backend: Backend
    :param model: Model type to be calibrated and stored in backend.
    :type model: _BaseExCorrectModel
    :param tolerance: Model can be perturbed by exact values too close to 0, this parameter sets
    an allowed distance between exact value and 0.
    :type tolerance: float
    """

    def cdr_calibration_task(
        obj,
        calibration_results: List[List[Tuple[QubitPauliOperator, QubitPauliOperator]]],
    ) -> Tuple[bool]:
        """
        For each experiments calibration results, calibrates a model of model type and stores it in backend.

        :param calibration_results: A list of noisy and noiseless expectation values for each
        experiment.
        :type calibration_results: List[List[Tuple[QubitPauliOperator, QubitPauliOperator]]]

        :return: A bool confirming characteriastion is complete
        :rtype: Tuple[bool]
        """
        counter = 0
        for calibration in calibration_results:
            # dict from QubitPauliString to Tuple[List[float], List[float]]
            # facilitates characteriastion of different QubitPauliStrings
            # for different experiments
            noisy_char_dict: Dict[QubitPauliString, List[float]] = dict()
            exact_char_dict: Dict[QubitPauliString, List[float]] = dict()

            for qpo_pair in calibration:
                noisy_qpo = qpo_pair[0]
                exact_qpo = qpo_pair[1]

                # go through strings in operator
                for key in noisy_qpo._dict:
                    # make sure keys are present (don't initialise at start incase indexing missing)
                    if abs(exact_qpo[key]) > tolerance:
                        if key not in noisy_char_dict:
                            noisy_char_dict[key] = list()
                        if key not in exact_char_dict:
                            exact_char_dict[key] = list()
                        if key not in exact_qpo._dict:
                            raise ValueError(
                                "Given key in calibration task for Clifford Data Regression should be present in exact and noisy characterisation results."
                            )

                        noisy_char_dict[key].append(float(noisy_qpo._dict[key]))
                        exact_char_dict[key].append(float(exact_qpo._dict[key]))
            if backend.backend_info is None:
                raise ValueError("Backend has no backend_info attribute.")

            backend.backend_info.misc["CDR_" + str(counter)] = dict()
            # for each qubit pauli string in operator, add model for calibrating
            for key in noisy_char_dict:
                model.calibrate(noisy_char_dict[key], exact_char_dict[key])
                backend.backend_info.misc["CDR_" + str(counter)][key] = copy.copy(model)
            counter += 1

        return (True,)

    return MitTask(
        _label="CDRCalibrate",
        _n_in_wires=1,
        _n_out_wires=1,
        _method=cdr_calibration_task,
    )


def cdr_correction_task_gen(backend: Backend) -> MitTask:
    """
    For each QubitPauliOperator passed, corrects the given expectation for each
    internal QubitPauliString via some pre-characterised mode.

    :param backend: Backend object where characterisation is stored.
    :type backend: Backend
    """

    def cdr_correction_task(
        obj,
        calibration_complete: bool,
        noisy_expectation: List[QubitPauliOperator],
    ) -> Tuple[List[QubitPauliOperator]]:
        """
        :param noisy_expectation: QubitPauliOperator objects from some experiment
        which are to be corrected from some predefined characteriastion.
        :type noisy_expectation: List[QubitPauliOperator]
        :param calibration_complete: bool passed to method once calibration task has completed.
        :type calibration_complete: bool

        """
        if backend.backend_info is None:
            raise ValueError("Backend has no backend_info attribute.")

        corrected_expectations = []
        for i in range(len(noisy_expectation)):
            char_string = "CDR_" + str(i)
            if char_string not in backend.backend_info.misc:
                raise RuntimeError(
                    "CDR characterisation not stored in backend.charactersation attribute."
                )
            models = backend.backend_info.misc[char_string]
            new_qpo_dict = dict()
            for qps in noisy_expectation[i]._dict:
                if qps in models:
                    new_qpo_dict[qps] = cast(
                        Union[int, float, complex],
                        models[qps].correct(float(noisy_expectation[i]._dict[qps])),
                    )
                else:
                    new_qpo_dict[qps] = cast(
                        Union[int, float, complex], noisy_expectation[i]._dict[qps]
                    )
            corrected_expectations.append(QubitPauliOperator(new_qpo_dict))

        return (corrected_expectations,)

    return MitTask(
        _label="CDRCorrect",
        _n_in_wires=2,
        _n_out_wires=1,
        _method=cdr_correction_task,
    )

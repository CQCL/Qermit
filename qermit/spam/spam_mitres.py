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


from qermit import (
    MitRes,
    TaskGraph,
)
from qermit.taskgraph import (
    backend_compile_circuit_shots_task_gen,
)
from qermit.spam.full_spam_correction import (
    gen_full_tomography_spam_characterisation_task,
    gen_full_tomography_spam_circuits_task,
    gen_full_tomography_spam_correction_task,
    CorrectionMethod,
    gen_get_bit_maps_task,
)
from pytket.backends import Backend
from typing import List
from pytket.circuit import Node  # type: ignore
from copy import copy


def gen_FullyCorrelated_SPAM_MitRes(
    backend: Backend, calibration_shots: int, correlations: List[List[Node]], **kwargs
) -> MitRes:
    """
    Produces a MitRes object for performing SPAM correction with subsets of fully
    correlated device nodes specified by correlations. Requires 2^n circuits
    to fully characterise, where n is the size of the largest sublist of correlated nodes.

    :param backend: Default Backend characterisation and experiment are executed on.
    :type backend: Backend
    :param calibration_shots: Number of shots required for each characterisation circuit
    :type calibration_shots: int
    :param correlations: Each sublist of Node corresponds to some set of fully correlated nodes.
    :type correlations: List[List[Node]]
    """
    _mitres_spam_calib = copy(
        kwargs.get("calibration_mitres", MitRes(backend, _label="SPAMCalibration"))
    )
    _mitres_spam_calib._label = "SPAMCalibration"

    _task_graph_spam_calib = TaskGraph().from_TaskGraph(_mitres_spam_calib)

    _mitres_spam_correction = copy(
        kwargs.get("correction_mitres", MitRes(backend, _label="SPAMCorrection"))
    )
    _mitres_spam_correction._label = "SPAMCorrection"

    _task_graph_spam_correction = TaskGraph().from_TaskGraph(_mitres_spam_correction)

    _task_graph_spam_correction.add_wire()
    _task_graph_spam_correction.prepend(gen_get_bit_maps_task())
    _task_graph_spam_correction.prepend(
        backend_compile_circuit_shots_task_gen(
            backend, kwargs.get("optimisation_level", 1)
        )
    )

    _task_graph_spam_calib.add_wire()
    _task_graph_spam_calib.append(
        gen_full_tomography_spam_characterisation_task(backend, correlations)
    )

    _task_graph_spam_correction.parallel(_task_graph_spam_calib)
    _task_graph_spam_correction.prepend(
        gen_full_tomography_spam_circuits_task(backend, calibration_shots, correlations)
    )
    _task_graph_spam_correction.append(
        gen_full_tomography_spam_correction_task(
            backend, kwargs.get("correction_method", CorrectionMethod.Invert)
        )
    )
    return MitRes(backend).from_TaskGraph(_task_graph_spam_correction)


def gen_UnCorrelated_SPAM_MitRes(
    backend: Backend, calibration_shots: int, **kwargs
) -> MitRes:
    """
    Produces a MitRes object for performing SPAM correction with no correlated nodes.
    Requires 2 circuits to characterise device.

    :param backend: Default Backend characterisation and experiment are executed on.
    :type backend: Backend
    :param calibration_shots: Number of shots required for each characterisation circuit
    :type calibration_shots: int
    """
    if backend.backend_info is None:
        raise ValueError("Backend has no backend_info attribute.")
    correlations = [[n] for n in backend.backend_info.architecture.nodes]
    return gen_FullyCorrelated_SPAM_MitRes(
        backend, calibration_shots, correlations, **kwargs
    )

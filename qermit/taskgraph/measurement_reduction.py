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


from typing import List, Tuple, Dict
from .mitex import MitEx
from .mittask import MitTask, ObservableExperiment
from .utils import MeasurementCircuit

import copy

from pytket.backends import Backend
from pytket.transform import CXConfigType  # type: ignore
from pytket.partition import (  # type: ignore
    PauliPartitionStrat,
    measurement_reduction,
    GraphColourMethod,
)
from pytket import Bit
from pytket.pauli import QubitPauliString  # type: ignore


def measurement_reduction_task_gen(
    strategy: PauliPartitionStrat, method: GraphColourMethod, cx_config: CXConfigType
) -> MitTask:
    """
    Uses measurement reduction techniques available in the pytket.partition module to reduce
    the number of measurement circuits required to measure all expectations for an observable.

    :param strategy: Measurement reduction strategy used in measurement_reduction method
    :type strategy: PauliPartitionStrat
    :param method: Graph colouring method used in measurement_reduction method
    :type method:  GraphColourMethod
    :param cx_config: Configuration of CX gates for diagonlisation methods.
    :type cx_config: CXConfigType

    :return: MitTask object, taking and return same write type, but with added
        measurement circuits to ObservableTracker.
    :rtype: MitTask
    """

    def task(
        obj,
        measurement_wires: List[ObservableExperiment],
    ) -> Tuple[List[ObservableExperiment]]:
        new_wires = []
        # for each experiment
        for measurement_wire in measurement_wires:
            # circ is ansatz circuit which measurement circuits are appended to
            circ = measurement_wire[0][0]
            # symbols dict object
            symbols = measurement_wire[0][2]
            # ObservableTracker object for given experiment
            tracker = measurement_wire[1]
            # use measurement reduction method to get MeasurementSetup object
            # with reduced number of measurement circuits
            new_setup = measurement_reduction(
                list(tracker._qubit_pauli_operator._dict.keys()),
                strat=strategy,
                method=method,
                cx_config=cx_config,
            )
            # attach measurement configurations from MeasurementSetup to
            # copies of AnsatzCircuit
            # new measurement circuits
            measurement_circuits = []
            for measurement_circuit in new_setup.measurement_circs:
                full_circ = circ.copy()
                full_circ.append(measurement_circuit.copy())
                measurement_circuits.append(MeasurementCircuit(full_circ, symbols))
            # convert MeasurementBitMap objects to MeasurementInfo for ObservableTracker
            adder_info: Dict[
                int, List[Tuple[QubitPauliString, List[Bit], bool]]
            ] = dict()
            for i in range(len(measurement_circuits)):
                adder_info[i] = list()

            for qps, value in new_setup.results.items():
                for v in value:
                    circ_index = v.circ_index
                    bits = [Bit(i) for i in v.bits]
                    invert = v.invert
                    adder_info[circ_index].append((qps, bits, invert))

            for key in adder_info:
                tracker.add_measurement_circuit(
                    measurement_circuits[key], adder_info[key]
                )

            new_wires.append(
                ObservableExperiment(
                    AnsatzCircuit=measurement_wire[0], ObservableTracker=tracker
                )
            )
        return (new_wires,)

    return MitTask(
        _label="MeasurementReduction", _n_in_wires=1, _n_out_wires=1, _method=task
    )


def gen_MeasurementReduction_MitEx(backend: Backend, **kwargs) -> MitEx:
    """
    Returns a MitEx object that produces measurement circuits via Measurement Reduction
    methods.

    :param backend: Backend experiment is built around.
    :type backend: Backend
    :key mitex: MitEx object measurement reduction task is prepended to.

    :return: MitEx object performing Measurement reduction
    :rtype: MitEx
    """
    # mitex object to built measurement reduction
    _mitex = copy.copy(
        kwargs.get("mitex", MitEx(backend, _label="MeasurementReductionMitEx"))
    )
    _mitex._label = "MeasurementReductionMitEx"
    _strategy = kwargs.get("strategy", PauliPartitionStrat.NonConflictingSets)
    _method = kwargs.get("method", GraphColourMethod.Lazy)
    _cx_config = kwargs.get("cx_config", CXConfigType.Snake)
    # add measurement reduction task to start of method
    _mitex.prepend(measurement_reduction_task_gen(_strategy, _method, _cx_config))
    return _mitex

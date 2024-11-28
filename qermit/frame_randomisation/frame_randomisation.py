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


import operator
from copy import copy
from enum import Enum
from functools import reduce
from math import ceil
from typing import Counter, List, Sequence, Tuple, cast

from pytket import Bit, Circuit, OpType
from pytket.backends import Backend
from pytket.backends.backendresult import BackendResult
from pytket.passes import AutoRebase
from pytket.tailoring import PauliFrameRandomisation, UniversalFrameRandomisation
from pytket.utils.outcomearray import OutcomeArray

from qermit import (
    CircuitShots,
    MitRes,
    MitTask,
)
from qermit.taskgraph.mitex import backend_compile_circuit_shots_task_gen


class FrameRandomisation(Enum):
    @staticmethod
    def PauliFrameRandomisation(
        circuit: Circuit, shots: int, samples: int
    ) -> List[CircuitShots]:
        """
        Uses the pytket.tailoring method PauliFrameRandomisation  to return a list of
        circuits corresponding to instances of frame randomisation for the input circuit.

        :param circuit: Circuit to have frame randomisation applied.
        :param shots: Number of shots of each frame randomisation circuit to take.
        :param samples: Number of frame randomisation instances to sample.
        """
        pfr = PauliFrameRandomisation()
        pfr_shots = ceil(shots / samples)
        AutoRebase({OpType.CX, OpType.Rz, OpType.H, OpType.S}).apply(circuit)
        pfr_circuits = pfr.sample_circuits(circuit, samples)
        return [CircuitShots(Circuit=c, Shots=pfr_shots) for c in pfr_circuits]

    @staticmethod
    def UniversalFrameRandomisation(
        circuit: Circuit, shots: int, samples: int
    ) -> List[CircuitShots]:
        """
        Uses the pytket.tailoring method UniversalFrameRandomisation  to return a list of
        circuits corresponding to instances of frame randomisation for the input circuit.
        UniversalFrameRandomisation rebases the circuit to a gate set such that
        the whole circuit is a single cycle for frame randomisation. This is possible with
        the additional noise assumption on top of the regular Frame Randomisation assumptions
        that Rz(-x) and Rz(x) incur similar noise for any angle x.

        :param circuit: Circuit to have frame randomisation applied.
        :param shots: Number of shots of each frame randomisation circuit to take.
        :param samples: Number of frame randomisation instances to sample.
        """
        ufr = UniversalFrameRandomisation()
        ufr_shots = ceil(shots / samples)

        AutoRebase({OpType.CX, OpType.Rz, OpType.H}).apply(circuit)
        ufr_circuits = ufr.sample_circuits(circuit, samples)
        return [CircuitShots(Circuit=c, Shots=ufr_shots) for c in ufr_circuits]


def frame_randomisation_circuits_task_gen(
    samples: int, _fr_type: FrameRandomisation
) -> MitTask:
    """
    Returns a MitTask object that produces Frame Randomisation circuits for some wire of experiment circuits.

    :param samples: Number of samples of frame randomisation circuits to take for each circuit in wire.
    """

    def task(obj, circs_shots: List[CircuitShots]) -> Tuple[List[CircuitShots]]:
        """
        :param circ_shots: A list of tuple of circuits and shots. Each circuit has frame randomisation applied.
        The number of shots of each frame randomisation circuit is ceil(shots/samples)

        :return: Frame Randomisation circutis
        """

        all_fr_circs_shots = []
        for circ, shots in circs_shots:
            fr_circshots = _fr_type(circ, shots, samples)  # type: ignore
            # frame randomisation type not callable as no default enum
            all_fr_circs_shots.extend(fr_circshots)
        return (all_fr_circs_shots,)

    return MitTask(
        _label="GenerateFrameRandomisationCircuits",
        _n_in_wires=1,
        _n_out_wires=1,
        _method=task,
    )


def frame_randomisation_result_task_gen(samples: int) -> MitTask:
    """
    Returns a MitTask object that sequentially collates samples number of BackendResult objects into single
    BackendResult object. These collated BackendResult objects should include all frame experiments for a single
    original circuit.

    :param samples: Number of frame randomisation instances created in the first place, used to suitably
        collate results.
    """

    def task(
        obj,
        all_fr_results: List[BackendResult],
    ) -> Tuple[List[BackendResult]]:
        """
        :param all_fr_results: A list of BackendResult objects for all frame randomisations for all experiment circuits.

        :return: Collated BackendResult objects, reflecting frame randomisation procedure
        """
        chunked_results = [
            all_fr_results[i : i + samples]
            for i in range(0, len(all_fr_results), samples)
        ]
        results = []
        for chunk in chunked_results:
            all_counts = [result.get_counts() for result in chunk]
            combined_counts = reduce(operator.add, all_counts)
            outcome_array = {
                OutcomeArray.from_readouts([key]): val
                for key, val in combined_counts.items()
            }
            outcome_counts = Counter(outcome_array)
            results.append(
                BackendResult(
                    counts=outcome_counts, c_bits=cast(Sequence[Bit], chunk[0].c_bits)
                )
            )
        return (results,)

    return MitTask(
        _label="CollateFrameRandomisationResults",
        _n_in_wires=1,
        _n_out_wires=1,
        _method=task,
    )


def gen_Frame_Randomisation_MitRes(backend: Backend, samples: int, **kwargs) -> MitRes:
    """
    Produces a MitRes object that applies FrameRandomisation techniques to experiment circuits.

    :param backend: Backend which experiments are default run through.
    :param samples: Number of Frame Randomisation instances sampled in Frame Randomisation.

    :key mitres: MitRes object FrameRandomisation MitRes built around if given.
    :key frame_randomisation: FrameRandomisation Enum passed to specify method used.
        Default set to FrameRandomisation.UniversalFrameRandomisation.
    """
    _mitres = copy(
        kwargs.get("mitres", MitRes(backend, _label="FrameRandomisationMitRes"))
    )
    _fr_type = copy(
        kwargs.get(
            "frame_randomisation", FrameRandomisation.UniversalFrameRandomisation
        )
    )
    _mitres.prepend(backend_compile_circuit_shots_task_gen(backend))
    _mitres.prepend(frame_randomisation_circuits_task_gen(samples, _fr_type))
    _mitres.append(
        frame_randomisation_result_task_gen(samples),
    )
    for n in _mitres._task_graph.nodes:
        if hasattr(n, "_label"):
            n._label = "FR" + n._label
    return _mitres

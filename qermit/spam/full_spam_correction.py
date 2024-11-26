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


from typing import Dict, List, Set, Tuple

from pytket import Bit, Circuit, OpType, Qubit
from pytket.backends import Backend
from pytket.backends.backendresult import BackendResult
from pytket.unit_id import Node

from qermit import (
    CircuitShots,
    MitTask,
)
from qermit.spam.full_transition_tomography import (
    CorrectionMethod,
    StateInfo,
    calculate_correlation_matrices,
    correct_transition_noise,
    get_full_transition_tomography_circuits,
)


def gen_full_tomography_spam_circuits_task(
    backend: Backend, shots: int, qubit_subsets: List[List[Node]]
) -> MitTask:
    """Generate MitTask for calibration circuits according to the specified correlation and given backend.

    :param backend: Backend on which the experiments are run.
    :param qubit_subsets: A list of lists of correlated Nodes of a `Device`.
        Qubits within the same list are assumed to only have SPAM errors correlated
        with each other. Thus to allow SPAM errors between all qubits you should
        provide a single list.  The qubits in `qubit_subsets` must be nodes in the
        backend's associated `Device`.
    :param shots: An int corresponding to the number of shots of each calibration circuit required.
    :return: A MitTask object, requiring 1 List[CircuitShots] wire and returning (List[CircuitShots], List[StateInfo])
        corresponding to Calibration Circuits and corresponding states.
    """

    def task(
        obj, wire: List[CircuitShots]
    ) -> Tuple[List[CircuitShots], List[CircuitShots], List[StateInfo]]:
        if "FullCorrelatedSpamCorrection" in obj.characterisation:
            # check correlations distance
            if (
                obj.characterisation["FullCorrelatedSpamCorrection"].CorrelatedNodes
                == qubit_subsets
            ):
                return (wire, [], [])

        process_circuit = Circuit(
            len([qb for subset in qubit_subsets for qb in subset])
        )
        tomo_circuit_states = get_full_transition_tomography_circuits(
            process_circuit, backend, qubit_subsets
        )

        tomo_circuit_shots = [
            CircuitShots(Circuit=c, Shots=shots) for c in tomo_circuit_states[0]
        ]
        return (wire, tomo_circuit_shots, tomo_circuit_states[1])

    return MitTask(
        _label="SPAMFullTomographyCircuits", _n_in_wires=1, _n_out_wires=3, _method=task
    )


def gen_full_tomography_spam_characterisation_task(
    qubit_subsets: List[List[Node]],
) -> MitTask:
    """
    Uses results from device for characterisation circuits to characterise transition matrices
    for different qubit subsets and stores them in backend.

    :param qubit_subsets: Subsets of qubits in backend corresponding to different correlated subsets.
    """

    def task(
        obj, results: List[BackendResult], state_infos: List[StateInfo]
    ) -> Tuple[bool]:
        """
        :param results: Results from characterisation circuits run on backend.
        :param state_infos: Corresponding state prepared in the circuit run for each result and qubit to bit map.

        :return: bool confirming characterisation complete.
        """
        if len(results) != len(state_infos):
            raise ValueError(
                "SPAM Characterisation requires the same number of prepared states and results."
            )
        if len(results) > 0:
            obj.characterisation["FullCorrelatedSpamCorrection"] = (
                calculate_correlation_matrices(results, state_infos, qubit_subsets)
            )

        return (True,)

    return MitTask(
        _label="SPAMFullCharacterisationCircuits",
        _n_in_wires=2,
        _n_out_wires=1,
        _method=task,
    )


def gen_full_tomography_spam_correction_task(corr_method: CorrectionMethod) -> MitTask:
    """
    Uses characterisation result held in backend to correct for SPAM noise in passed
    BackendResult objects. Method used to invert SPAM characteriastion matrices
    and correct results given by CorrectionMethod enum.

    :param corr_method: Method used to invert matrices and correct results.
    """

    def task(
        obj,
        results: List[BackendResult],
        bit_qb_maps: List[Tuple[Dict[Qubit, Bit], Dict[Bit, Qubit]]],
        characterised: bool,
    ) -> Tuple[List[BackendResult]]:
        """
        :param results: Results from experiment circuits run on backend.
        :param bit_qb_maps: Map between Bits measurement outcomes are assigned to and Qubits in each experiment Circuit. Separate dicts for
        results end of and mid-circuit measurements.

        :return: Corrected Results
        """
        if "FullCorrelatedSpamCorrection" in obj.characterisation:
            char = obj.characterisation["FullCorrelatedSpamCorrection"]
        else:
            raise ValueError(
                "'FullCorrelatedSpamCorrection' not characterised for Backend."
            )
        if len(results) != len(bit_qb_maps):
            raise ValueError(
                "Number of experiment results and Qubit to Bit maps do not match."
            )
        corrected_results = [
            correct_transition_noise(r, qbm, char, corr_method)
            for r, qbm in zip(results, bit_qb_maps)
        ]
        return (corrected_results,)

    return MitTask(
        _label="SPAMFullCorrection", _n_in_wires=3, _n_out_wires=1, _method=task
    )


def get_mid_circuit_measure_map(
    circuit: Circuit, used_bits: Set[Bit] = set()
) -> Dict[Bit, Qubit]:
    """
    For each circuit, gets all Measure commands and uses them to construct a dictionary
    between Bit and Qubit.

    :param circuit: Circuit to get dict between Bit measured and Qubit measured on.

    :return: A dict between Bit measured and Qubit measured on.
    """
    bit_to_qubit_map = dict()
    for mc in circuit.commands_of_type(OpType.Measure):
        bit = mc.bits[0]
        if bit not in used_bits:
            bit_to_qubit_map[bit] = mc.qubits[0]
    return bit_to_qubit_map


def gen_get_bit_maps_task() -> MitTask:
    """
    Returns a task that takes a list of circuits and returns the circuits, and a map betwen
    each circuit bit and the qubit it is measured on.
    """

    def task(
        obj, circuit_shots: List[CircuitShots]
    ) -> Tuple[List[CircuitShots], List[Tuple[Dict[Qubit, Bit], Dict[Bit, Qubit]]]]:
        """
        :param circuits: Circuits to retrieve bit maps from.

        :return: A tuple comprising the original circuits, and each circuits bit map.
        """
        bq_maps = []
        for c in circuit_shots:
            qb_map = c[0].qubit_to_bit_map
            # if condition met, implies that mid circuit measurement has ocurred and not accounted for
            # in this case, iterate through circuit commands to get Qubits for all Bits
            if len(qb_map) != len(c[0].bits):
                bq_maps.append(
                    (qb_map, get_mid_circuit_measure_map(c[0], set(qb_map.values())))
                )
            else:
                # else, just invert map for later correction
                bq_maps.append((qb_map, dict()))
        return (circuit_shots, bq_maps)

    return MitTask(
        _label="GetBitQubitMaps", _n_in_wires=1, _n_out_wires=2, _method=task
    )

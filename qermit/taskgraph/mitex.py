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


import inspect
from copy import deepcopy
from typing import List, OrderedDict, Tuple, Union, cast

import networkx as nx  # type: ignore
from pytket import Bit, Circuit
from pytket.backends import Backend
from pytket.backends.backendresult import BackendResult
from pytket.pauli import Pauli, QubitPauliString
from pytket.utils import QubitPauliOperator

from .mitres import (
    MitRes,
    backend_compile_circuit_shots_task_gen,
    gen_shot_split_MitRes,
)
from .mittask import (
    AnsatzCircuit,
    CircuitShots,
    IOTask,
    MitTask,
    ObservableExperiment,
    Wire,
)
from .task_graph import TaskGraph
from .utils import MeasurementCircuit, MeasurementInfo, ObservableTracker, SymbolsDict


def gen_compiled_shot_split_MitRes(
    backend: Backend, max_shots: int, optimisation_level: int = 1
) -> MitRes:
    """
    Returns a shot splitting MitRes object with a compilation task prepended that compiles
    circuit wires via backend.compile_circuit. Optimisation level can be optionally
    set as defined by backend.compile_circuit.

    :param backend: Backend with circuits are compiled for.
    :param max_shots: The maximum number of shots that each job should request.
    :param optimisation_level: Sets options in compile_circuit method

    :return: shot splitting MitRes object with compilation task prepended.
    """
    mr = gen_shot_split_MitRes(backend, max_shots)
    mr.prepend(backend_compile_circuit_shots_task_gen(backend, optimisation_level))
    return mr


def gen_compiled_MitRes(backend: Backend, optimisation_level: int = 1) -> MitRes:
    """
    Returns a MitRes object with a compilation task prepended that compiles
    circuit wires via backend.compile_circuit. Optimisaion level can be optionally
    set as defined by backend.compile_circuit.

    :param backend: Backend with circuits are compiled for.
    :param optimisation_level: Sets options in compile_circuit method

    :return: MitRes object with compilation task prepended.
    """
    mr = MitRes(backend)
    mr.prepend(backend_compile_circuit_shots_task_gen(backend, optimisation_level))
    return mr


def get_basic_measurement_circuit(
    string: QubitPauliString,
) -> Tuple[Circuit, MeasurementInfo]:
    """
    Given a Qubit Pauli String, returns a circuit for measuring qubits in given basis
    via changing of basis through quantum gates.

    :param string: Qubit Pauli String to be measured

    :return: Measurement circuit for appending on some ansatz
    """
    measurement_circuit = Circuit()
    measured_qbs = []
    for qb, p in string.map.items():
        if p == Pauli.I:
            continue
        measured_qbs.append(qb)
        measurement_circuit.add_qubit(qb)
        if p == Pauli.X:
            measurement_circuit.H(qb)
        elif p == Pauli.Y:
            measurement_circuit.Rx(0.5, qb)
    bits = []
    for b_idx, qb in enumerate(measured_qbs):
        unit = Bit(b_idx)
        bits.append(unit)
        measurement_circuit.add_bit(unit, False)
        measurement_circuit.Measure(qb, unit)
    return (measurement_circuit, (string, bits, False))


def filter_observable_tracker_task_gen() -> MitTask:
    """
    Generates basic (changing measurement basis via Pauli gates) MeasurementCircuit
    for every QubitPauliString passed that has no Measurementcircuit in ObservableTracker object passed on wire.

    :return: Pure function that adds MeasurementCircuit objects to ObservableTracker.
    """

    def task(
        obj,
        measurement_wires: List[ObservableExperiment],
    ) -> Tuple[List[List[CircuitShots]], List[ObservableTracker]]:
        """
        :param measurement_wires: Wires containing Circuit information and Observable information
        :returns: A modified ObservableTracker object, and a List of CircuitShots for each Observable measured
        """

        output_circuits = []
        output_trackers = []
        for measurement_wire in list(measurement_wires):
            # set variables
            ansatz_circuit = measurement_wire[0]
            circuit = ansatz_circuit[0]
            shots = ansatz_circuit[1]
            symbols = ansatz_circuit[2]
            observable_tracker = measurement_wire[1]

            non_existant_qubits = (
                observable_tracker.qubit_pauli_operator.all_qubits.difference(
                    set(circuit.qubits)
                )
            )
            # TODO: In a future refactor, this check should be done
            # inside ObservableExperiment.
            if len(non_existant_qubits) > 0:
                raise Exception(
                    f"ObservableTracker qubits {non_existant_qubits} are not found in circuit."
                )

            # first make sure all observable has some measurement circuit
            strings_for_circuits = observable_tracker.get_empty_strings()
            for string in strings_for_circuits:
                circ = circuit.copy()
                # tuple, first entry is measurement circuit for appending
                # second entry is MeasurementInfo for deriving expectation
                measurement_circuit = get_basic_measurement_circuit(string)
                circ.append(measurement_circuit[0])
                # add new circuit to observable tracker
                observable_tracker.add_measurement_circuit(
                    MeasurementCircuit(circ, symbols), [measurement_circuit[1]]
                )

            # retrieve all measurement circuits, substitute symbols
            output_circuits.append(
                [
                    CircuitShots(Circuit=mc.get_parametric_circuit(), Shots=shots)
                    for mc in observable_tracker.measurement_circuits
                ]
            )
            output_trackers.append(observable_tracker)
        return (output_circuits, output_trackers)

    return MitTask(
        _label="FilterObservableTracker", _n_in_wires=1, _n_out_wires=2, _method=task
    )


def collate_circuit_shots_task_gen() -> MitTask:
    """
    Each wire contains a single experiment with its own List of Circuits to run.
    To improve parallelisation (i.e. reduce queueing time), these lists are collated
    and queued for a device at the same time.
    """

    def task(
        obj, circuit_wires: List[List[CircuitShots]]
    ) -> Tuple[List[CircuitShots], List[int]]:
        """
        :param circuit_wires: Different lists of Circuit + Shots for different experiments

        :return: Different experiment circuits collated on one wire, and the sequential number of
        results for each experiment on a second wire.
        """
        collated_circuitshots = []
        lengths = []
        for wire in circuit_wires:
            lengths.append(len(wire))
            collated_circuitshots.extend(wire)
        return (collated_circuitshots, lengths)

    return MitTask(
        _label="CollateExperimentCircuits", _n_in_wires=1, _n_out_wires=2, _method=task
    )


def split_results_task_gen() -> MitTask:
    """
    Returned list of Results from MitRes object are for multiple experiments.
    This method generates a task that converts the list of results into a list
    of list of results, wherein each list is for a different MitEx experiment.
    """

    def task(
        obj, results: List[BackendResult], experiment_sizes: List[int]
    ) -> Tuple[List[List[BackendResult]]]:
        """
        :param results: All results returned from MitRes object
        :param experiment_sizes: The ordered number of results required for each experiment

        :return: All results split up into sublists for each MitEx experiment
        """
        lower_bound = 0
        split_results = []
        for size in experiment_sizes:
            upper_bound = lower_bound + size
            split_results.append(results[lower_bound:upper_bound])
            lower_bound = upper_bound
        return (split_results,)

    return MitTask(_label="SplitResults", _n_in_wires=2, _n_out_wires=1, _method=task)


def get_expectations_task_gen() -> MitTask:
    """
    Passes each set of experiment results to corresponding ObservableTracker method, returning
    a QubitPauliOperator object containing expectation values from Results multiplied by coefficients.
    """

    def task(
        obj, all_results: List[List[BackendResult]], trackers: List[ObservableTracker]
    ) -> Tuple[List[QubitPauliOperator]]:
        """
        :param all_results: All Results from MitRes split into sublists, one for each Observable experiment.
        :param trackers: All ObservableTrackers defining experiments passed to MitEx.

        :return: Each experiments expectation results in a QubitPauliOperator
        """
        if len(all_results) != len(trackers):
            raise ValueError(
                "{} results and {} observable trackers. Values should be identical.".format(
                    len(all_results), len(trackers)
                )
            )
        output_qpos = [
            observable.get_expectations(results)
            for observable, results in zip(trackers, all_results)
        ]
        for observable in trackers:
            observable.clear()
        return (output_qpos,)

    return MitTask(
        _label="GenerateExpectations", _n_in_wires=2, _n_out_wires=1, _method=task
    )


class MitEx(TaskGraph):
    """
    A TaskGraph extension for mitigation of expectation values for individual QubitPauliStrings
            contained in some ObservableTracker/QubitPauliOperator.
    """

    def __init__(self, backend: Backend, _label: str = "MitEx", **kwargs) -> None:
        """
        MitEx objects are defined by the backend object experiments are run through.
        However, as experiments run through some MitRes object, kwargs
        can be used to run through any mitres of choice.

        :param backend: Pytket backend default constructor which tasks are generated from.
        :param label: Name for identification of MitEx object.
        :key mitres: MitEx object experiments are run through
        """
        # set member variables
        self._label = _label
        self._n_wires = 1
        self.G = None
        self.characterisation: dict = {}

        # start building default MitEx task graph
        self._task_graph = nx.MultiDiGraph()
        self._i, self._o = IOTask.Input, IOTask.Output

        # if requested, all data is held in cache and can be accessed after running
        self._cache: OrderedDict[str, Tuple[MitTask, List[Wire]]] = OrderedDict()

        # add edge from input to filtering task to generate measurement circuits
        filter_observable_tracker_task = filter_observable_tracker_task_gen()
        self._task_graph.add_edge(
            self._i, filter_observable_tracker_task, key=(0, 0), data=None
        )

        collate_circuit_shots_task = collate_circuit_shots_task_gen()
        self._task_graph.add_edge(
            filter_observable_tracker_task,
            collate_circuit_shots_task,
            key=(0, 0),
            data=None,
        )

        # if mitres isn't defined, build around a mitres which compiles circuits
        _mitres = kwargs.get("mitres", gen_compiled_MitRes(backend))
        self._task_graph.add_edge(
            collate_circuit_shots_task, _mitres, key=(0, 0), data=None
        )

        split_results_task = split_results_task_gen()
        self._task_graph.add_edge(_mitres, split_results_task, key=(0, 0), data=None)
        self._task_graph.add_edge(
            collate_circuit_shots_task, split_results_task, key=(1, 1), data=None
        )

        get_expectations_task = get_expectations_task_gen()
        self._task_graph.add_edge(
            split_results_task, get_expectations_task, key=(0, 0), data=None
        )
        self._task_graph.add_edge(
            filter_observable_tracker_task, get_expectations_task, key=(1, 1), data=None
        )
        self._task_graph.add_edge(get_expectations_task, self._o, key=(0, 0), data=None)

    def check_prepend_wires(self, task: Union[MitTask, TaskGraph]) -> bool:
        """
        Confirms that the number of out wires the passed task has is equal
        to the number of out wires from the input, and that the number
        of in wires the passed task has is 1. Also checks that the
        task.run attribute argument is List[ObservableExperiment] and that its
        return type is Tuple[List[ObservableExperiment]].

        :param task: MitTask or TaskGraph object for checking wire numbers of.

        :return: True if task is suitably for prepending, False if not.
        """
        sig = inspect.signature(task.run)
        params = list(sig.parameters.values())
        return (
            (task.n_out_wires == self.n_in_wires)
            and (task.n_in_wires == 1)
            and (len(params) == 1)
            and (params[0].annotation == List[ObservableExperiment])
            and (sig.return_annotation == Tuple[List[ObservableExperiment]])
        )

    def check_append_wires(self, task: Union[MitTask, TaskGraph]) -> bool:
        """
        Confirms that the number of in wires the passed task has is equal
        to the number of in wires to the output, and that the number
        of out wires the passed task has is 1. Also checks that the
        task.run attribute argument is List[QubitPauliOperator] and that its
        return type is Tuple[List[QubitPauliOperator]].

        :param task: MitTask or TaskGraph object for checking wire numbers of.

        :return: True if task is suitably for appending, False if not.
        """
        sig = inspect.signature(task.run)
        params = list(sig.parameters.values())
        return (
            (task.n_in_wires == self.n_out_wires)
            and (task.n_out_wires == 1)
            and (len(params) == 1)
            and (params[0].annotation == List[QubitPauliOperator])
            and (sig.return_annotation == Tuple[List[QubitPauliOperator]])
        )

    def __str__(self):
        return f"<MitEx::{self._label}>"

    def __call__(  # type: ignore[override]
        self,
        experiment_wires: List[List[ObservableExperiment]],
        cache: bool = False,
        characterisation: dict = {},
    ) -> Tuple[List[QubitPauliOperator]]:
        return cast(
            Tuple[List[QubitPauliOperator]],
            super().run(
                cast(List[Wire], experiment_wires),
                cache=cache,
                characterisation=characterisation,
            ),
        )

    def from_TaskGraph(self, task_graph: TaskGraph):
        """
        Returns a MitEx object from a TaskGraph object.

        :param task_graph: TaskGraph object to copy tasks from.

        :return: Copied TaskGraph as MitEx
        """
        if task_graph.n_in_wires != 1 or task_graph.n_out_wires != 1:
            raise TypeError(
                "Type signature of passed task_graph.run method does not equal MitEx.run type signature. Number of in and out wires does not match."
            )
        # can index as previous check means there should only be one edge
        input_parameters = list(
            inspect.signature(
                list(task_graph._task_graph.out_edges(task_graph._i))[0][1].run
            ).parameters.values()
        )
        if (
            len(input_parameters) != 1
            and input_parameters[0].annotation != List[ObservableExperiment]
        ):
            raise TypeError(
                "Type signature of passed task_graph.run method does not equal MitEx.run type signature. First MitTask in graph should expect a single argument of List[ObservableExperiment], but expects {}.".format(
                    input_parameters
                )
            )

        # can index as previous check means there should only be one edge
        return_annotation = inspect.signature(
            list(task_graph._task_graph.in_edges(task_graph._o))[0][0].run
        ).return_annotation
        if return_annotation != Tuple[List[QubitPauliOperator]]:
            raise TypeError(
                "Type signature of passed task_graph.run method does not equal MitEx.run type signature. Last MitTask in"
                "task graph should return Tuple[List[QubitPauliOperator]], but returns {}.".format(
                    return_annotation
                )
            )
        self._task_graph = deepcopy(task_graph._task_graph)
        self._label = task_graph._label
        return self

    def parallel(self, task: Union[MitTask, "TaskGraph"]):
        """
        Requests to add new MitTask/TaskGraph to TaskGraph object in parallel.
        Not permitted for MitEx, raises TypeError.

        :param task: New task to be added in parallel.
        """
        raise TypeError("MitEx.parallel forbidden.")

    def add_n_wires(self, num_wires: int):
        """
        Requests to add num_wires number of edges between the input vertex
        and output vertex, with no type restrictions. Not permitted for MitEx,
        raises TypeError.

        :param num_wires: Number of edges to add between input and output vertices.
        """
        raise TypeError("MitEx.add_n_wires forbidden.")

    def add_wire(self):
        """
        Requests to add a single edge between the input vertex and output vertex.
        Not permitted for MitEx, raises TypeError.
        """
        raise TypeError("MitEx.add_wire forbidden.")

    def run(  # type: ignore[override]
        self,
        mitex_wires: List[ObservableExperiment],
        cache: bool = False,
        characterisation: dict = {},
    ) -> List[QubitPauliOperator]:
        """
        Overloaded run method.
        A single observable experiment is defined by a Tuple containg an Ansatz
        Circuit object and an ObservableTracker object.
        An AnsatzCircuit is a tuple containing a Circuit without measures (the ansatz circuit), the number
        of shots to be taken of each Measurement Circuit later run and a SymbolsDict object
        holding a dictionary between Circuit Symbolics (if present) and values
        for substituting them with when running measurement circuits (i.e. parameters).
        It is useful to keep parameters as symbolics until measurement circuits are executed
        as some mitigation methods benefit from being able to run differently parameterised circuits.
        An ObservableTracker object is defined by a QubitPauliOperator defining the observable of interest
        (a dictionary between QubitPauliStrings and coefficients), and later stores MeasurementCircuit
        objects for running on devices.

        :param mitex_wires: Each Tuple pertains to a different Observable measuring experiment, and contains
            the minimum amount of information to run an Mitigated Experiment for calculating observables.

        :return: Observable experiment results as QubitPauliOperator, where values are expectations.
        """
        return self([mitex_wires], cache, characterisation)[0]

    def run_basic(
        self, mitex_wires: List[Tuple[CircuitShots, QubitPauliOperator]]
    ) -> List[QubitPauliOperator]:
        """
        Additional run option for MitEx which simplifies the arguments required for basic experiments.
        An experiment is defined here as just the ansatz circuit, number of shots and the operator being measured.
        This method converts these into the arguments for using run, calls run and then returns the
        desired operator.

        :param mitex_wires: Each tuple pertains to a different basic observable measuring experiment.

        :return: Observable experiment results as QubitPauliOperator, where values are expectations.
        """
        run_wires = [
            ObservableExperiment(
                AnsatzCircuit(m[0].Circuit, m[0].Shots, SymbolsDict()),
                ObservableTracker(m[1]),
            )
            for m in mitex_wires
        ]
        return self.run(run_wires)

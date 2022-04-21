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

from .mittask import (
    MitTask,
    IOTask,
    CircuitShots,
    Wire,
)
from typing import List, Tuple, Union, OrderedDict, cast, Sequence
from .task_graph import TaskGraph
from pytket.backends import Backend, ResultHandle
from pytket.backends.backendresult import BackendResult
import networkx as nx  # type: ignore
import inspect
from copy import deepcopy
from itertools import repeat
from pytket.utils.outcomearray import OutcomeArray
from pytket import Bit
import numpy as np


def backend_compile_circuit_shots_task_gen(
    backend: Backend, optimisation_level: int = 1
) -> MitTask:
    """
    For each circuit in passed List[CircuitShots] argument, pass
    circuit to compile_circuit method of given backend with given optimisation level.
    Returns new wire wherein each circuit has been compiled.

    :param backend: Backend object from which compile circuit method is called
    :type backend: Backend
    :param optimisation_level: Optimisation level of backend called.
    :type optimmisation_level: int
    """

    def task(obj, circ_shots: List[CircuitShots]) -> Tuple[List[CircuitShots]]:
        return (
            [
                CircuitShots(
                    backend.get_compiled_circuit(
                        cs.Circuit, optimisation_level=optimisation_level
                    ),
                    cs.Shots,
                )
                for cs in circ_shots
            ],
        )

    return MitTask(
        _label="CompileCircuitShots", _n_in_wires=1, _n_out_wires=1, _method=task
    )


def backend_handle_task_gen(backend: Backend) -> MitTask:
    """
    Passes every tuple of Circuit and Shots to the backend object MitTask
    is defined by, returning a handle for later retrieving results for each circuit.
    If different numbers of shots are passed, each circuit is run with the maximum number of shots.

    :param backend: Backend circuits are run through.
    :type backend: Backend
    :return: Pure function that adds passes circuits to backend and gets handles.
    :rtype: MitTask
    """

    def task(obj, circuit_wires: List[CircuitShots]) -> Tuple[List[ResultHandle]]:
        """
        :param circuit_wires: Circuits to be run on backend, number of shots to run of each.
        :type circuit_wires: List[CircuitShots]

        :return: ResultHandles from process_circuits method.
        :rtype: Tuple[List[ResultHandle]]
        """

        if len(circuit_wires) != 0:
            circs, shots = map(list, zip(*circuit_wires))

            results = backend.process_circuits(
                circs, n_shots=cast(Sequence[int], shots)
            )

            return (results,)
        else:
            return ([],)

    return MitTask(
        _label="CircuitsToHandles", _n_in_wires=1, _n_out_wires=1, _method=task
    )


def backend_res_task_gen(backend: Backend) -> MitTask:
    """
    For each ResultHandle passed to task, retrieves a BackendResult object from
    the backend the task is defined by.

    :param backend: backend holding results for handles.
    :type backend: Backend
    """

    def task(obj, handles: List[ResultHandle]) -> Tuple[List[BackendResult]]:

        results = backend.get_results(handles)

        return (results,)
        """
        :param handles: ResultHandle objects previously produced from backend.
        :type handles: List[ResultHandle]

        :return: For each ResultHandle in handles, a BackendResult object retrieved from backend.
        :rtype: Tuple[List[BackendResult]]
        """

    return MitTask(
        _label="HandlesToResults", _n_in_wires=1, _n_out_wires=1, _method=task
    )


class MitRes(TaskGraph):
    """
    A TaskGraph extension of mitigation of counts/shots for individual circuits.
    """

    def __init__(
        self,
        backend: Backend,
        _label: str = "MitRes",
    ) -> None:
        """
        MitRes objects are defined by the backend objects all circuits are executed on.


        :param backend: Pytket backend default constructor which tasks are generated from.
        :type backend: Backend
        :param label: Name for identification of MitRes object.
        :type label: str
        """
        # set member variables
        self._label = _label
        self.G = None

        # default constructor runs all circuits through passed Backend
        self._task_graph = nx.MultiDiGraph()

        # if requested, all data is held in cache and can be accessed after running
        self._cache: OrderedDict[str, Tuple[MitTask, List[Wire]]] = OrderedDict()

        c2h = backend_handle_task_gen(backend)
        h2r = backend_res_task_gen(backend)

        self._i, self._o = IOTask.Input, IOTask.Output
        self._task_graph.add_edge(self._i, c2h, key=(0, 0), data=None)
        self._task_graph.add_edge(c2h, h2r, key=(0, 0), data=None)
        self._task_graph.add_edge(h2r, self._o, key=(0, 0), data=None)

    def check_prepend_wires(self, task: Union[MitTask, "TaskGraph"]) -> bool:
        """
        Confirms that the number of out wires the passed task has is equal
        to the number of out wires from the input, and that the number
        of in wires the passed task has is 1. Also checks that the
        task.run attribute argument is List[CircuitShots] and that its
        return type is Tuple[List[CircuitShots]].

        :param task: MitTask or TaskGraph object for checking wire numbers of.
        :type task: Union[MitTask, Taskgraph]

        :return: True if task is suitable for prepending, False if not.
        :rtype: bool
        """
        sig = inspect.signature(task.run)
        params = list(sig.parameters.values())
        return (
            (task.n_out_wires == self.n_in_wires)
            and (task.n_in_wires == 1)
            and (len(params) == 1)
            and (params[0].annotation == List[CircuitShots])
            and (sig.return_annotation == Tuple[List[CircuitShots]])
        )

    def check_append_wires(self, task: Union[MitTask, "TaskGraph"]) -> bool:
        """
        Confirms that the number of in wires the passed task has is equal
        to the number of in wires to the output, and that the number
        of out wires the passed task has is 1. Also checks that the
        task.run attribute argument is List[BackendResult] and that its
        return type is Tuple[List[BackendResult]].

        :param task: MitTask or TaskGraph object for checking wire numbers of.
        :type task: Union[MitTask, Taskgraph]

        :return: True if task is suitable for apppending, False if not.
        :rtype: bool
        """
        sig = inspect.signature(task.run)
        params = list(sig.parameters.values())
        return (
            (task.n_in_wires == self.n_out_wires)
            and (task.n_out_wires == 1)
            and (len(params) == 1)
            and (params[0].annotation == List[BackendResult])
            and (sig.return_annotation == Tuple[List[BackendResult]])
        )

    def __str__(self) -> str:
        return f"<MitRes::{self._label}>"

    def __call__(self, circuits_wire: List[List[CircuitShots]]) -> Tuple[List[BackendResult]]:  # type: ignore[override]
        return cast(
            Tuple[List[BackendResult]], super().run(cast(List[Wire], circuits_wire))
        )

    def from_TaskGraph(self, task_graph: TaskGraph):
        """
        Returns a MitRes object from a TaskGraph object.

        :param task_graph: TaskGraph object to copy tasks from.
        :type task_graph: TaskGraph
        :return: Copied TaskGraph as MitRes
        :rtype: MitRes
        """
        if task_graph.n_in_wires != 1 or task_graph.n_out_wires != 1:
            raise TypeError(
                "Type signature of passed task_graph.run method does not equal MitRun.run type signature. Number of in and out wires does not match."
            )
        # can index as previous check means there should only be one edge
        input_parameters = list(
            inspect.signature(
                list(task_graph._task_graph.out_edges(task_graph._i))[0][1].run
            ).parameters.values()
        )
        if (
            len(input_parameters) != 1
            and input_parameters[0].annotation != List[CircuitShots]
        ):
            raise TypeError(
                "Type signature of passed task_graph.run method does not equal MitRes.run type signature. First MitTask in graph should expect a single argument of List[Circuitshots], but expects {}.".format(
                    input_parameters
                )
            )

        # can index as previous check means there should only be one edge
        return_annotation = inspect.signature(
            list(task_graph._task_graph.in_edges(task_graph._o))[0][0].run
        ).return_annotation
        if return_annotation != Tuple[List[BackendResult]]:
            raise TypeError(
                "Type signature of passed task_graph.run method does not equal MitRes.run type signature. Last MitTask in"
                "task graph should return Tuple[List[BackendResult]], but returns {}.".format(
                    return_annotation
                )
            )
        self._task_graph = deepcopy(task_graph._task_graph)
        self._label = task_graph._label
        return self

    def parallel(self, task: Union[MitTask, "TaskGraph"]):
        """
        Requests to add new MitTask/TaskGraph to TaskGraph object in parallel.
        Not permitted for MitRes, raises TypeError.

        :param task: New task to be added in parallel.
        :type task: MitTask
        """
        raise TypeError("MitRes.parallel forbidden.")

    def add_n_wires(self, num_wires: int):
        """
        Requests to add num_wires number of edges between the input vertex
        and output vertex, with no type restrictions. Not permitted for MitRes,
        raises TypeError.


        :param num_wires: Number of edges to add between input and output vertices.
        :type num_wires: int
        """
        raise TypeError("MitRes.add_n_wires forbidden.")

    def add_wire(self):
        """
        Requests to add a single edge between the input vertex and output vertex.
        Not permitted for MitRes, raises TypeError.
        """
        raise TypeError("MitRes.add_wire forbidden.")

    def run(self, circuit_shots: List[CircuitShots]) -> List[BackendResult]:  # type: ignore[override]
        """
        Overloaded run method from TaskGraph class to add type checking.
        A single experiment is defined by a Tuple containing a circuit to be run
        on some backend and the number of shots to take of said circuit.
        For each combination of Circuit and shots run returns a BackendResult object
        containing counts/shots.

        :param circuit_shotss: Each tuple in circuit_wires contains a Circuit to run on
            internal backends and the number of shots to take of said circuit.
        :type circuit_shots: List[CircuitShots]

        :return: A BackendResult object for each combination of circuit and shots.
        :rtype: List[BackendResult]
        """
        return self([circuit_shots])[0]


def split_shots_task_gen(max_shots: int) -> MitTask:
    """
    When the number of shots requested is higher than max_shots, this task
    splits the jobs into several individual jobs, where the total number of
    shots matches that initially requested.

    :param max_shots: The maximum number of shots per job.
    :type max_shots: int
    :return: A task dividing the job into several smaller ones.
    :rtype: MitTask
    """

    def task(
        obj, circuit_wires: List[CircuitShots]
    ) -> Tuple[List[CircuitShots], List[int]]:
        """
        This task divides a job into several smaller ones, if the number of
        shots requested is larger then the maximum allowed.

        :param circuit_wires: A list of the circuits to be run, and the number
        of shots to be taken.
        :type circuit_wires: List[CircuitShots]
        :return: A new list of circuits, some of which repeat with a smaller
        number of shots.
        :rtype: Tuple[List[CircuitShots], List[int]]
        """

        # A new list of circuits and shots
        split_circuit_wires = []
        # and index of which circuits in the new list correspond to which in
        # the original list. This is to say all circuits in the new list
        # assigned an index i by this index list correspond to original
        # circuit i.
        split_index: List[int] = []

        # For each circuit, check if the number of shots requested is greater
        # then the maximum, and divide the job into several smaller ones if
        # this is the case.
        for i, circ_shots in enumerate(circuit_wires):
            div_val = circ_shots.Shots // max_shots
            # Divide into jobs of size max_shots
            if circ_shots.Shots > max_shots:
                split_circuit_wires.extend(
                    [
                        CircuitShots(circ_shots.Circuit, max_shots)
                        for _ in range(div_val)
                    ]
                )
            # Add remaining shots
            if circ_shots.Shots % max_shots > 0:
                split_circuit_wires.append(
                    CircuitShots(circ_shots.Circuit, circ_shots.Shots % max_shots)
                )
            # Add new job indexes
            split_index.extend(
                repeat(i, div_val + min(1, circ_shots.Shots % max_shots))
            )

        return (split_circuit_wires, split_index)

    return MitTask(_label="SplitShots", _n_in_wires=1, _n_out_wires=2, _method=task)


def group_shots_task_gen() -> MitTask:
    """
    Returns task which groups together jobs that correspond to the same
    circuit. Original larger job could have been split by task returned
    by split_shots_task_gen.

    :return: Task merging jobs which correspond to the same circuit.
    :rtype: MitTask
    """

    def task(
        obj, results_wires: List[BackendResult], split_index: List[int]
    ) -> Tuple[List[BackendResult]]:
        """
        Task which merges jobs which correspond to the same circuit.

        :param results_wires: A list of results, some of which correspond to
        the same circuit.
        :type results_wires: List[BackendResult]
        :param split_index: A list of indexes identifying which results
        correspond to which circuit. Those results with index i correspond to
        the circuit i.
        :type split_index: List[int]
        :return: A new list of results, where results which correspond to the
        same circuit have been grouped together.
        :rtype: Tuple[List[BackendResult]]
        """

        # A new list of results where some results have been merged.
        merged_results = []

        # A list of lists. Each internal list contains results that correspond
        # to the circuit of the same index.
        grouped_results = [
            [result for result, i in zip(results_wires, split_index) if i == j]
            for j in range(split_index[-1] + 1)
        ]

        # For each circuit, combine the results.
        for chunk in grouped_results:
            # Concatenate shots from all results
            combined_shots = np.concatenate([result.get_shots() for result in chunk])
            outcome_array = OutcomeArray.from_readouts(combined_shots)
            merged_results.append(
                BackendResult(
                    shots=outcome_array, c_bits=cast(Sequence[Bit], chunk[0].c_bits)
                )
            )

        return (merged_results,)

    return MitTask(_label="MergeShots", _n_in_wires=2, _n_out_wires=1, _method=task)


def gen_shot_split_MitRes(backend: Backend, max_shots: int) -> MitRes:
    """
    Wraps a mitres in tasks which ensures shots requested by individual
    jobs never exceed max_shots. It does so by splitting the shots over several
    jobs if necessary.

    :param backend: The backend on which to run the jobs.
    :type backend: Backend
    :param max_shots: The maximum number of shots that each job should request.
    :type max_shots: int
    :return: A MitRes object.
    :rtype: MitRes
    """

    _experiment_mitres = MitRes(backend=backend)

    _experiment_taskgraph = TaskGraph().from_TaskGraph(_experiment_mitres)

    _experiment_taskgraph.add_wire()

    # Prepend splitting task.
    split_shots = split_shots_task_gen(max_shots)
    _experiment_taskgraph.prepend(split_shots)

    # Append merging task.
    merge_shots = group_shots_task_gen()
    _experiment_taskgraph.append(merge_shots)

    return MitRes(backend).from_TaskGraph(_experiment_taskgraph)

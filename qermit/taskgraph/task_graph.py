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

from copy import copy, deepcopy
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, List, OrderedDict, Tuple, Union, cast

import networkx as nx  # type: ignore

from .graphviz import _taskgraph_to_graphviz
from .mittask import (
    IOTask,
    MitTask,
    Wire,
)

if TYPE_CHECKING:
    import graphviz as gv  # type: ignore


class TaskGraph:
    """
    The TaskGraph class stores a networkx graph where vertices
    are pure functions or tasks, and edges hold data.
    In the TaskGraph class these tasks and edges have no
    type restrictions, though for the run method to be succesful, the
    types of ports edges are attached to must match.

    :param _label: Name for identification of TaskGraph object.
    """

    def __init__(
        self,
        _label: str = "TaskGraph",
    ) -> None:
        # set member variables
        self._label = _label
        self.G = None
        self.characterisation: dict = {}
        # default constructor runs all circuits through passed Backend
        self._task_graph = nx.MultiDiGraph()

        self._i, self._o = IOTask.Input, IOTask.Output
        self._task_graph.add_edge(self._i, self._o, key=(0, 0), data=None)

        # if requested, all data is held in cache and can be accessed after running
        self._cache: OrderedDict[str, Tuple[MitTask, List[Wire]]] = OrderedDict()

    def from_TaskGraph(self, task_graph: "TaskGraph"):
        """
        Returns a new TaskGraph object from another TaskGraph object.

        :param task_graph: TaskGraph object to copy tasks from.

        :return: Copied TaskGraph
        """
        self._task_graph = deepcopy(task_graph._task_graph)
        self._label = task_graph._label
        self.characterisation = task_graph.characterisation
        return self

    @property
    def tasks(self) -> List[MitTask]:
        """
        Returns a list of all tasks with both input and output ports
        in the TaskGraph.
        """
        return list(self._task_graph)[2:]

    def __call__(self, input_wires: List[Wire]) -> Tuple[List[Wire]]:
        return self.run(input_wires)

    @property
    def label(self) -> str:
        return self._label

    def get_characterisation(self) -> dict:
        return self.characterisation

    def update_characterisation(self, characterisation: dict):
        self.characterisation.update(characterisation)

    def set_characterisation(self, characterisation: dict):
        self.characterisation = characterisation

    @property
    def n_in_wires(self) -> int:
        """
        The number of in wires to a TaskGraph object is defined as the number
        of out edges from the Input Vertex, as when called, a TaskGraph object
        calls the run method which stores input arguments as data on Input vertex
        output edges.
        """
        return len(self._task_graph.out_edges(self._i))

    @property
    def n_out_wires(self) -> int:
        """
        The number of out wires to a TaskGraph object is defined as the number
        of in edges to the Input Vertex, as when called, a TaskGraph object
        calls the run method which after running all tasks, returns
        the data on input edges to the Output Vertex as a tuple.
        """
        return len(self._task_graph.in_edges(self._o))

    def check_prepend_wires(self, task: Union[MitTask, "TaskGraph"]) -> bool:
        """
        Confirms that the number of out wires of the proposed task to prepend to the
        internal task_graph attribute matches the number of in wires to the graph.

        :param task: Wrapped pure function to prepend to graph

        :return: True if prepend permitted
        """
        return task.n_out_wires == self.n_in_wires

    def check_append_wires(self, task: Union[MitTask, "TaskGraph"]) -> bool:
        """
        Confirms that the number of in wires of the proposed task to append to the
        internal task_graph attribute matches the number of out wires to the graph.

        :param task: Wrapped pure function to append to graph

        :return: True if append permitted
        """
        return task.n_in_wires == self.n_out_wires

    def __str__(self):
        return f"<TaskGraph::{self._label}>"

    def __repr__(self):
        return str(self)

    def add_n_wires(self, num_wires: int):
        """
        Adds num_wires number of edges between the input vertex
        and output vertex, with no type restrictions.

        :param num_wires: Number of edges to add between input and output vertices.
        """
        for _ in range(num_wires):
            in_port = len(self._task_graph.out_edges(self._i, data=True))
            out_port = len(self._task_graph.in_edges(self._o, data=True))
            self._task_graph.add_edge(
                self._i, self._o, key=(in_port, out_port), data=None
            )

    def add_wire(self):
        """
        Adds a single edge between the input vertex and output vertex.
        """
        self.add_n_wires(1)

    # Add news task to start of TaskGraph
    def prepend(self, task: Union[MitTask, "TaskGraph"]):
        """
        Inserts new task to the start of TaskGraph._task_graph.
        All out edges from the Input vertex are wired as out edges from the task in the same port ordering (types must match).
        New edges also added from the Input vertex to the task (any type permitted), ports ordered in arguments order.

        :param task: New task to be prepended.
        """
        assert self.check_prepend_wires(task)
        # It's possible a single generated MitTask object could be used in different TaskGraph objects
        # via prepend which may lead to a task address expecting input wires from different graphs
        # use of copy here prevents this and task graph generation is not the bottleneck in running mitigation
        # schemes so fine
        task_copy = copy(task)

        for i, edge in enumerate(list(self._task_graph.out_edges(self._i, keys=True))):
            self._task_graph.add_edge(
                task_copy, edge[1], key=(i, edge[2][1]), data=None
            )
            self._task_graph.remove_edge(edge[0], edge[1])

        for port in range(task_copy.n_in_wires):
            self._task_graph.add_edge(self._i, task_copy, key=(port, port), data=None)

    def append(self, task: Union[MitTask, "TaskGraph"]):
        """
        Inserts new task to end of TaskGraph._task_graph.
        All in edges to Output vertex are wired as in edges to task in same port ordering (types must match).
        New edges added from task to Output vertex (any type permitted), ports ordered in arguments order.

        :param task: New task to be appended.
        """
        assert self.check_append_wires(task)
        # It's possible a single generated MitTask object could be used in different TaskGraph objects
        # via append which may lead to a task address expecting input wires from different graphs
        # use of copy here prevents this and task graph generation is not the bottleneck in running mitigation
        # schemes so fine
        task_copy = copy(task)
        for edge in list(self._task_graph.in_edges(self._o, keys=True)):
            self._task_graph.add_edge(edge[0], task_copy, key=edge[2], data=None)

            self._task_graph.remove_edge(edge[0], edge[1])

        for port in range(task_copy.n_out_wires):
            self._task_graph.add_edge(task_copy, self._o, key=(port, port), data=None)

    def decompose_TaskGraph_nodes(self):
        """
        For each node in self._task_graph, if node is a TaskGraph object, substitutes that node
        with the _task_graph structure held inside the node.
        """
        check_for_decompose = True
        while check_for_decompose:
            # get all nodes and iterate through them
            check_for_decompose = False
            node_list = list(nx.topological_sort(self._task_graph))
            for task in node_list:
                # => TaskGraph object with _task_graph attribute for decomposition
                if hasattr(task, "_task_graph"):
                    # relabel task names for ease of viewing wit visualisation methods
                    for sub_task in list(task._task_graph.nodes):
                        # in practice only IOTask
                        if hasattr(sub_task, "_label"):
                            sub_task._label = task._label + sub_task._label

                    task_in_edges = list(self._task_graph.in_edges(task, keys=True))
                    task_out_edges = list(self._task_graph.out_edges(task, keys=True))

                    task_input_out_edges = list(
                        task._task_graph.out_edges(task._i, keys=True)
                    )
                    task_output_in_edges = list(
                        task._task_graph.in_edges(task._o, keys=True)
                    )

                    if (
                        len(
                            set(task_input_out_edges).intersection(
                                set(task_output_in_edges)
                            )
                        )
                        > 0
                    ):
                        raise ValueError(
                            "Decomposition of TaskGraph node {}, not permitted: TaskGraph to be decomposed has edge between input and output vertices.".format(
                                task
                            )
                        )

                    # These two cases imply faulty TaskGraph construction
                    # Note that this check is only made as this is necessary constraint for decomposing TaskGraph nodes
                    # Faulty construction should be caught at construction of TaskGraph object, including types
                    if len(task_in_edges) != len(task_input_out_edges):
                        raise TypeError(
                            "Decomposition of TaskGraph node {} not permitted: node "
                            "expects {} input wires but receives {}.".format(
                                task, len(task_input_out_edges), len(task_in_edges)
                            )
                        )
                    if len(task_out_edges) != len(task_output_in_edges):
                        raise TypeError(
                            "Decomposition of TaskGraph node {} not permitted: task_graph "
                            "expects {} output wires but node returns {}.".format(
                                task, len(task_output_in_edges), len(task_out_edges)
                            )
                        )

                    # remove all input and output edges from task._task_graph
                    # remove all input and output edges from self._task_graph task
                    # replace in_edges to task in self._task_graph with in_edges to first tasks in task
                    for outside_edge, inside_edge in zip(
                        task_in_edges, task_input_out_edges
                    ):
                        task._task_graph.remove_edge(inside_edge[0], inside_edge[1])
                        self._task_graph.remove_edge(outside_edge[0], outside_edge[1])
                        self._task_graph.add_edge(
                            outside_edge[0],
                            inside_edge[1],
                            key=(outside_edge[2][0], inside_edge[2][1]),
                            data=None,
                        )

                    for outside_edge, inside_edge in zip(
                        task_out_edges, task_output_in_edges
                    ):
                        task._task_graph.remove_edge(inside_edge[0], inside_edge[1])
                        self._task_graph.remove_edge(outside_edge[0], outside_edge[1])
                        self._task_graph.add_edge(
                            inside_edge[0],
                            outside_edge[1],
                            key=(inside_edge[2][0], outside_edge[2][1]),
                            data=None,
                        )

                    # add all remaining edges, filling the rest of the subsituted graph
                    self._task_graph.add_edges_from(task._task_graph.edges)
                    self._task_graph.remove_node(task)
                    check_for_decompose = True
                    break

    def parallel(self, task: Union[MitTask, "TaskGraph"]):
        """
        Adds new MitTask/TaskGraph to TaskGraph object in parallel. All task in edges wired as out edges from Input vertex. All task out_Edges wired as in edges to Output Vertex.

        :param task: New task to be added in parallel.
        """
        task = copy(task)
        base_n_input_outs = len(self._task_graph.out_edges(self._i))
        for port in range(task.n_in_wires):
            self._task_graph.add_edge(
                self._i,
                task,
                key=(base_n_input_outs + port, port),
                data=None,
            )
        base_n_output_ins = len(self._task_graph.in_edges(self._o))
        for port in range(task.n_out_wires):
            self._task_graph.add_edge(
                task,
                self._o,
                key=(port, base_n_output_ins + port),
                data=None,
            )

    def run(
        self, input_wires: List[Wire], cache: bool = False, characterisation: dict = {}
    ) -> Tuple[List[Wire]]:
        """
        Each task in TaskGraph is a pure function that produces output data
        from input data to some specification. Data is stored on edges of the
        internal _task_graph object.
        The run method first assigns each wire in the list to an output edge
        of the input vertex. If more wires are passed than there are edges, wire
        information is not assigned to some edge. If less wires are passed then there are edges,
        some edges will have no data and later computation will likely fail.
        Nodes (holding either MitTask or TaskGraph callable objects) on the graph undergo a topological sort to order them, and are then
        executed sequentially.
        To be executed, data from in edges to a task are passed arguments to the tasks
        _method, and data returned from method are assigned to out edges from a task.
        This process is repeated until all tasks are run, at which point all in edges
        to the output vertex wil have data on, with each data set returned in a tuple.

        :param input_wires: Each Wire holds information assigned as data to an output edge
            from the input vertex of the _task_graph.
        :param cache: If True each Tasks output data is stored in an OrderedDict with the
            Task._label attribute as its key.


        :return: Data from input edges to output vertex, assigned as wires.
        """
        for edge, wire in zip(
            self._task_graph.out_edges(self._i, data=True), input_wires
        ):
            edge[2]["data"] = wire

        # topological_sort fixes any dependency issues so can iterate and assume
        # input wires all realised before a task is reached
        node_list = list(nx.topological_sort(self._task_graph))

        self.characterisation.update(characterisation)
        # clear cache of held data if required
        # also check that all mittask label are unique else dict will fail
        if cache:
            unique_labels = set()
            for task in node_list:
                if task not in (self._i, self._o):
                    unique_labels.add(task._label)
            if len(unique_labels) != len(self._task_graph) - 2:
                raise ValueError(
                    "Cache can't store all information as not all MitTask labels are unique."
                )
            else:
                self._cache.clear()

        for task in node_list:
            # nothing to process
            if task in (self._i, self._o):
                continue
            task.characterisation.update(self.characterisation)
            # get all input data and store on inputs for task
            in_edges = self._task_graph.in_edges(task, data=True, keys=True)
            inputs = [None] * len(in_edges)
            for _, _, ports, i_data in in_edges:
                assert i_data["data"] is not None
                inputs[ports[1]] = i_data["data"]
            # run held task
            outputs = task(inputs)
            self.characterisation.update(task.characterisation)
            if cache:
                self._cache[task._label] = (task, outputs)
            # assign outputs ot out_edges of task
            out_edges = self._task_graph.out_edges(task, data=True, keys=True)
            assert len(out_edges) == len(outputs)
            for _, _, ports, o_data in out_edges:
                o_data["data"] = outputs[ports[0]]

        output_wire = [
            edge[2]["data"]
            for edge in list(self._task_graph.in_edges(self._o, data=True))
        ]
        return cast(Tuple[List[Wire]], tuple(output_wire))

    def get_cache(self) -> OrderedDict[str, Tuple[MitTask, List[Wire]]]:
        """
        :returns: Dictionary holding all output data from all MitTask.
            This is only full after run is called with the cache argument set
            to True. Keys are stored in graph topological order.
        """
        return self._cache

    def get_task_graph(self) -> "gv.Digraph":
        """
        Return a visual representation of the DAG as a graphviz object.

        :returns:   Representation of the DAG
        """
        return _taskgraph_to_graphviz(self._task_graph, None, self._label)

    def view_task_graph(self) -> None:
        """
        View the DAG.
        """
        G = self.get_task_graph()
        file = NamedTemporaryFile()
        G.view(file.name, quiet=True)

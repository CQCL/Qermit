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
    Wire,
)
import networkx as nx  # type: ignore
import graphviz as gv  # type: ignore
from typing import List, Union, Tuple, cast
import copy
from tempfile import NamedTemporaryFile


class TaskGraph:
    """
    The TaskGraph class stores a networkx graph where vertices
    are pure functions or tasks, and edges hold data.
    In the TaskGraph class these tasks and edges have no
    type restrictions, though for the run method to be succesful, the
    types of ports edges are attached to must match.

    :param _label: Name for identification of TaskGraph object.
    :type _label: str
    """

    def __init__(
        self,
        _label: str = "TaskGraph",
    ) -> None:
        # set member variables
        self._label = _label
        self.G = None

        # default constructor runs all circuits through passed Backend
        self._task_graph = nx.MultiDiGraph()

        self._i, self._o = IOTask.Input, IOTask.Output
        self._task_graph.add_edge(self._i, self._o, key=(0, 0), data=None)

    def from_TaskGraph(self, task_graph: "TaskGraph"):
        """
        Returns a new TaskGraph object from another TaskGraph object.

        :param task_graph: TaskGraph object to copy tasks from.
        :type task_graph: TaskGraph

        :return: Copied TaskGraph
        :rtype: TaskGraph
        """
        self._task_graph = copy.deepcopy(task_graph._task_graph)
        self._label = task_graph._label
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

    @property
    def n_in_wires(self):
        """
        The number of in wires to a TaskGraph object is defined as the number
        of out edges from the Input Vertex, as when called, a TaskGraph object
        calls the run method which stores input arguments as data on Input vertex
        output edges.
        """
        return len(self._task_graph.out_edges(self._i))

    @property
    def n_out_wires(self):
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
        :type task: Union[MitTask, "TaskGraph"]

        :return: True if prepend permitted
        :rtype: bool
        """
        return task.n_out_wires == self.n_in_wires

    def check_append_wires(self, task: Union[MitTask, "TaskGraph"]) -> bool:
        """
        Confirms that the number of in wires of the proposed task to append to the
        internal task_graph attribute matches the number of out wires to the graph.

        :param task: Wrapped pure function to append to graph
        :type task: Union[MitTask, "TaskGraph"]

        :return: True if append permitted
        :rtype: bool
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
        :type num_wires: int
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
        :type task: MitTask
        """
        assert self.check_prepend_wires(task)
        task_copy = copy.copy(task)

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
        :type task: MitTask
        """
        assert self.check_append_wires(task)
        task_copy = copy.copy(task)
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
        while check_for_decompose == True:
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

    # eat, yum,
    def sandwich(
        self,
        prepend_task: Union[MitTask, "TaskGraph"],
        append_task: Union[MitTask, "TaskGraph"],
    ):
        """
        Does TaskGraph.prepend(prepend_task) and TaskGraph.append(append_task). Archaic but delicious.

        :param prepend_task: New task to be prepended.
        :type prepend_task: MitTask
        :param append_task: New task to be appended.
        :type append_task: MitTask
        """
        self.prepend(prepend_task)
        self.append(append_task)

    def parallel(self, task: Union[MitTask, "TaskGraph"]):
        """
        Adds new MitTask/TaskGraph to TaskGraph object in parallel. All task in edges wired as out edges from Input vertex. All task out_Edges wired as in edges to Output Vertex.

        :param task: New task to be added in parallel.
        :type task: MitTask
        """
        task = copy.copy(task)
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

    def run(self, input_wires: List[Wire]) -> Tuple[List[Wire]]:
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
        :type input_wires: List[Wire]

        :return: Data from input edges to output vertex, assigned as wires.
        :rtype: Tuple[List[Wire]]

        """
        for edge, wire in zip(
            self._task_graph.out_edges(self._i, data=True), input_wires
        ):
            edge[2]["data"] = wire

        # TODO Parallelism an option here, async an option for doing so.
        node_list = list(nx.topological_sort(self._task_graph))

        for task in node_list:
            if task in (self._i, self._o):
                continue
            in_edges = self._task_graph.in_edges(task, data=True, keys=True)
            inputs = [None] * len(in_edges)
            for _, _, ports, i_data in in_edges:
                assert i_data["data"] is not None
                inputs[ports[1]] = i_data["data"]
            out_edges = self._task_graph.out_edges(task, data=True, keys=True)

            outputs = task(inputs)
            assert len(out_edges) == len(outputs)
            for _, _, ports, o_data in out_edges:
                o_data["data"] = outputs[ports[0]]

        output_wire = [
            edge[2]["data"]
            for edge in list(self._task_graph.in_edges(self._o, data=True))
        ]
        return cast(Tuple[List[Wire]], tuple(output_wire))

    def get_task_graph(self) -> gv.Digraph:
        """
        Return a visual representation of the DAG as a graphviz object.

        :returns:   Representation of the DAG
        :rtype:     graphviz.DiGraph
        """
        G = gv.Digraph(
            "MEME",
            strict=True,
        )

        G.attr(rankdir="LR", ranksep="0.3", nodesep="0.15", margin="0")
        wire_color = "red"
        task_color = "darkolivegreen3"
        io_color = "green"
        out_color = "black"
        in_color = "white"

        boundary_node_attr = {"fontname": "Courier", "fontsize": "8"}
        boundary_nodes = {self._i, self._o}

        with G.subgraph(name="cluster_input") as c:
            c.attr(rank="source")
            c.node_attr.update(shape="point", color=io_color)
            for i in range(len(self._task_graph.out_edges(self._i))):
                c.node(
                    name=str(((str(self._i) + "out").replace("::", "_"), i)),
                    xlabel="Input" + str(i),
                    **boundary_node_attr,
                )

        with G.subgraph(name="cluster_output") as c:
            c.attr(rank="sink")
            c.node_attr.update(shape="point", color=io_color)
            for i in range(len(self._task_graph.in_edges(self._o))):
                c.node(
                    name=str(((str(self._o) + "in").replace("::", "_"), i)),
                    xlabel="Output",
                    **boundary_node_attr,
                )

        node_cluster_attr = {
            "style": "rounded, filled",
            "color": task_color,
            "fontname": "Times-Roman",
            "fontsize": "10",
            "margin": "5",
            "lheight": "100",
        }
        in_port_node_attr = {
            "color": in_color,
            "shape": "point",
            "weight": "2",
            "fontname": "Helvetica",
            "fontsize": "8",
            "rank": "source",
        }
        out_port_node_attr = {
            "color": out_color,
            "shape": "point",
            "weight": "2",
            "fontname": "Helvetica",
            "fontsize": "8",
            "rank": "sink",
        }
        count = 0
        for node, ndata in self._task_graph.nodes.items():
            if node not in boundary_nodes:
                with G.subgraph(name="cluster_" + node._label + str(count)) as c:
                    count = count + 1
                    c.attr(label=str(node._label), **node_cluster_attr)

                    n_in_ports = self._task_graph.in_degree(node)
                    if n_in_ports == 1:
                        c.node(
                            name=str(((str(node) + "in").replace("::", "-"), 0)),
                            **in_port_node_attr,
                        )
                    else:
                        for i in range(n_in_ports):
                            c.node(
                                name=str(((str(node) + "in").replace("::", "-"), i)),
                                xlabel=str(i),
                                **in_port_node_attr,
                            )

                    n_out_ports = self._task_graph.out_degree(node)
                    if n_out_ports == 1:
                        c.node(
                            name=str(((str(node) + "out").replace("::", "-"), 0)),
                            **out_port_node_attr,
                        )
                    else:
                        for i in range(n_out_ports):
                            c.node(
                                name=str(((str(node) + "out").replace("::", "-"), i)),
                                xlabel=str(i),
                                **out_port_node_attr,
                            )

        edge_attr = {
            "weight": "2",
            "arrowhead": "vee",
            "arrowsize": "0.2",
            "headclip": "true",
            "tailclip": "true",
        }
        for edge, edata in self._task_graph.edges.items():
            src_node, tgt_node, _ = edge
            src_port = edge[2][0]
            tgt_port = edge[2][1]
            src_nodename = str(((str(src_node) + "out").replace("::", "-"), src_port))
            tgt_nodename = str(((str(tgt_node) + "in").replace("::", "-"), tgt_port))
            G.edge(src_nodename, tgt_nodename, color=wire_color, **edge_attr)
        self.G = G
        return G

    def view_task_graph(self) -> None:
        """
        View the DAG.
        """
        G = self.get_task_graph()
        file = NamedTemporaryFile()
        G.view(file.name, quiet=True)

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


from qermit import (  # type: ignore
    TaskGraph,
    MitTask,
)
from qermit.taskgraph import (  # type: ignore
    IOTask,
    duplicate_wire_task_gen,
)
import networkx as nx  # type: ignore
import pytest


def test_task_graph_constructor():
    tg = TaskGraph(_label="TestTaskGraph")
    # confirm basic properties about the graph
    assert str(tg) == "<TaskGraph::TestTaskGraph>"
    assert list(tg._task_graph)[0] == IOTask.Input
    assert list(tg._task_graph)[1] == IOTask.Output
    assert len(tg.tasks) == 0
    assert tg.n_in_wires == 1
    assert tg.n_out_wires == 1


def test_basic_task_graph_methods():
    tg = TaskGraph()
    assert tg.run([1, 2]) == (1,)

    def return_5(self, input):
        return (5,)

    task5 = MitTask(_label="Return5", _method=return_5, _n_in_wires=1, _n_out_wires=1)
    assert task5.n_in_wires == 1
    assert task5.n_out_wires == 1
    tg.prepend(task5)
    assert len(tg.tasks) == 1

    assert tg.tasks[0]._label == task5._label
    # just returns 1 as 5
    assert tg.run([1, 2]) == (5,)

    tg.add_n_wires(2)
    assert tg.n_out_wires == 3
    assert tg.n_in_wires == 3
    # second wire by passes return 5, so just returns 2
    # no information passed to third wire, so returns None
    assert tg.run([1, 2]) == (5, 2, None)

    # create a task to append that changes the number of wires
    # must have 3 input wires to match
    def return_multiply(self, input0, input1, input2):
        return (input0 * input1 * input2, input0 * input1)

    task_multiply = MitTask(
        _label="Multiply", _method=return_multiply, _n_in_wires=3, _n_out_wires=2
    )
    assert task_multiply.n_in_wires == 3
    assert task_multiply.n_out_wires == 2

    tg.append(task_multiply)
    assert tg.n_in_wires == 3
    assert tg.n_out_wires == 2
    assert len(tg.tasks) == 2
    assert tg.tasks[0]._label == task5._label
    assert tg.tasks[1]._label == task_multiply._label

    res = tg.run([1, 2, 3])
    assert res == (30, 10)

    # add a final task that should change the number of inputs to task graph
    duplicate_wire_task = duplicate_wire_task_gen(in_wires=1, duplicates=3)
    assert duplicate_wire_task._n_in_wires == 1
    assert duplicate_wire_task._n_out_wires == 3

    tg.prepend(duplicate_wire_task)
    assert tg.n_in_wires == 1
    assert tg.n_out_wires == 2
    assert len(tg.tasks) == 3
    assert tg.tasks[0]._label == task5._label
    assert tg.tasks[1]._label == task_multiply._label
    assert tg.tasks[2]._label == duplicate_wire_task._label
    # check results are expected also
    assert tg.run([10]) == (500, 50)

    tg.prepend(duplicate_wire_task_gen(1, 1))
    tg.append(duplicate_wire_task_gen(2, 2))
    assert len(tg.tasks) == 5
    assert tg.run([10]) == (500, 50, 500, 50)


def test_advanced_task_graph_methods():
    tg_0 = TaskGraph(_label="BaseTaskGraph")
    tg_0.add_wire()

    def return_add5s(self, input0, input1):
        return (input0 + 5, input1 + 5)

    task5 = MitTask(_label="Add5", _method=return_add5s, _n_in_wires=2, _n_out_wires=2)
    tg_0.prepend(task5)

    # add task graph object as task to taskgraph
    tg_1 = TaskGraph(_label="SecondTaskGraph")
    tg_1.add_n_wires(1)

    tg_1.prepend(tg_0)

    assert tg_1.run([2, 2]) == tg_0.run([2, 2])

    # add task graph object in parallel
    tg_2 = TaskGraph(_label="ParallelTG")
    tg_2.append(duplicate_wire_task_gen(1, 2))

    tg_1.parallel(tg_2)
    assert tg_1.run([2, 2, 3]) == (7, 7, 3, 3)
    assert len(tg_1.tasks) == 2

    tg_3 = TaskGraph().from_TaskGraph(tg_1)
    tg_3._label = "Copied"

    # confirm that graph copy works
    tg1_tasks = list(nx.topological_sort(tg_1._task_graph))
    tg3_tasks = list(nx.topological_sort(tg_3._task_graph))
    assert tg1_tasks[0] == tg3_tasks[0]
    assert tg1_tasks[1]._label == tg3_tasks[1]._label
    assert tg1_tasks[2]._label == tg3_tasks[2]._label
    assert tg1_tasks[3] == tg3_tasks[3]
    tg_1.parallel(tg_3)

    # confirm graph decompose works
    assert len(list(nx.topological_sort(tg_1._task_graph))) == 5
    tg_1.decompose_TaskGraph_nodes()
    assert len(list(nx.topological_sort(tg_1._task_graph))) == 6


def test_run_with_cache():
    tg = TaskGraph()

    def return_5(self, input):
        return (5,)

    task5_dummy0 = MitTask(
        _label="dummy", _method=return_5, _n_in_wires=1, _n_out_wires=1
    )
    tg.prepend(task5_dummy0)

    assert tg.run([1, 2], cache=False) == (5,)
    assert len(tg.get_cache()) == 0
    assert tg.run([1, 2], cache=True) == (5,)
    c = tg.get_cache()
    assert len(c) == 1
    assert c["dummy"][1] == (5,)
    task5_dummy1 = MitTask(
        _label="dummy", _method=return_5, _n_in_wires=1, _n_out_wires=1
    )
    tg.append(task5_dummy1)
    assert tg.run([1, 2], cache=False) == (5,)
    with pytest.raises(ValueError):
        tg.run([1, 2], cache=True)


if __name__ == "__main__":
    test_task_graph_constructor()
    test_basic_task_graph_methods()
    test_advanced_task_graph_methods()
    test_run_with_cache()

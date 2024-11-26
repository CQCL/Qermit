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


from collections import namedtuple
from enum import Enum
from types import MethodType
from typing import Callable, Dict, List, Optional, Union

from pytket import Bit, Circuit, Qubit
from pytket.backends import ResultHandle
from pytket.backends.backendresult import BackendResult
from pytket.utils import QubitPauliOperator


class IOTask(Enum):
    """
    Simple Node type for labelling the Input and Output Nodes to a TaskGraph object.
    """

    Input = 0
    Output = 1


CircuitShots = namedtuple("CircuitShots", ["Circuit", "Shots"])
AnsatzCircuit = namedtuple("AnsatzCircuit", ["Circuit", "Shots", "SymbolsDict"])
ObservableExperiment = namedtuple(
    "ObservableExperiment", ["AnsatzCircuit", "ObservableTracker"]
)

Wire = Union[
    CircuitShots,
    Circuit,
    BackendResult,
    ResultHandle,
    AnsatzCircuit,
    ObservableExperiment,
    int,
    float,
    bool,
    str,
    QubitPauliOperator,
    Dict[Qubit, Bit],
    Dict,
]


class MitTask:
    """
    An object a TaskGraph node is comprised of. A MitTask object
    is defined by the _method attribute, which holds a pure function
    that requires _n_in_wires input arguments and returns a Tuple of
    _n_out_wires objects. The object callable is defined as the
    _method attribute.

    :param _label: String to identify MitTask object by.
    :param _n_in_wires: Number of input arguments to _method attribute function.
    :param _n_out_wires: number of results in Tuple returned by _method attribute function.
    :param _method: Pure function executed when object called.

    :return: MitTask object for adding to TaskGraph.
    """

    def __init__(
        self,
        _label: str,
        _n_in_wires: int,
        _n_out_wires: int,
        _method: Optional[Callable] = None,
    ):
        self._label = _label
        self._n_in_wires = _n_in_wires
        self._n_out_wires = _n_out_wires
        if _method:
            self.run = MethodType(_method, self)
        self.characterisation: dict = {}

    @property
    def label(self) -> str:
        return self._label

    @property
    def n_in_wires(self):
        return self._n_in_wires

    @property
    def n_out_wires(self):
        return self._n_out_wires

    def __call__(self, input_wires: List[Wire]) -> List[Wire]:
        return self.run(*input_wires)

    def __str__(self):
        return f"<MitTask::{self._label}>"

    def __repr__(self):
        return str(self)


def duplicate_wire_task_gen(in_wires: int, duplicates: int) -> MitTask:
    """
    Generator for constructing a task that for each argument, corresponding
    to data from some input edge on graph, makes duplicates copies to be returned in Tuple
    (and later added to out edges of MitTask on TaskGraph).

    :param in_wires: Number of in edges to Task on TaskGraph.
    :param duplicates: Number of copies to take of each argument.
    """

    def task(obj, *args):
        if len(args) != in_wires:
            raise ValueError(
                "Task has "
                + str(len(args))
                + " input arguments but expects "
                + str(in_wires)
                + "."
            )
        return args * duplicates

    return MitTask(
        _label="Duplicate" + str(in_wires) + "Wires" + str(duplicates) + "Times",
        _n_in_wires=in_wires,
        _n_out_wires=in_wires * duplicates,
        _method=task,
    )

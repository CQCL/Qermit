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


"""
qermit provides tools for running and composing error-mitigation methods. Error
mitigation methods are split into two types, those which in some manner modify the set
of shots returned when running quantum circuits on quantum devices (MitRes), and those which
modify the expectation value of some observable (MitEx).
"""

from qermit.taskgraph.mitex import MitEx
from qermit.taskgraph.mitres import MitRes
from qermit.taskgraph.mittask import (
    AnsatzCircuit,
    CircuitShots,
    MitTask,
    ObservableExperiment,
)
from qermit.taskgraph.task_graph import TaskGraph
from qermit.taskgraph.utils import MeasurementCircuit, ObservableTracker, SymbolsDict

__path__ = __import__("pkgutil").extend_path(__path__, __name__)

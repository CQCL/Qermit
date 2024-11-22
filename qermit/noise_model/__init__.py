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
The noise_model module provides an assortment of noise modeling and simulation
tools for local testing.
"""

from .mock_quantinuum_backend import MockQuantinuumBackend
from .noise_model import (
    Direction,
    ErrorDistribution,
    LogicalErrorDistribution,
    NoiseModel,
)
from .pauli_error_transpile import PauliErrorTranspile
from .qermit_pauli import QermitPauli
from .transpiler_backend import TranspilerBackend

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


"""
The spam module defines tasks for completing SPAM Mitigation methods, and generator
functions for producing MitRes objects that automatically characterise and apply SPAM 
Mitigation methods to experiment results.
"""
from .spam_mitres import (
    gen_FullyCorrelated_SPAM_MitRes,
    gen_UnCorrelated_SPAM_MitRes,
    CorrectionMethod,
)
from .full_transition_tomography import FullCorrelatedNoiseCharacterisation

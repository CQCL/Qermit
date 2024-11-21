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
The spectral_filtering module provides generator methods for producing
MitEx objects that automatically characterise and correct device noise via
Spectral Filtering.
"""

from .signal_filter import SmallCoefficientSignalFilter
from .spectral_filtering import gen_spectral_filtering_MitEx

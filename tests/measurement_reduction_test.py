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
    SymbolsDict,
    ObservableTracker,
)
from qermit.taskgraph.measurement_reduction import (  # type: ignore
    gen_MeasurementReduction_MitEx,
    measurement_reduction_task_gen,
)
from pytket.extensions.qiskit import AerBackend  # type: ignore
from pytket.pauli import QubitPauliString, Pauli  # type: ignore
from pytket.utils import QubitPauliOperator
from pytket import Circuit, Qubit, Bit
from pytket.partition import (  # type: ignore
    PauliPartitionStrat,
    GraphColourMethod,
)
from pytket.transform import CXConfigType  # type: ignore


def test_measurement_reduction_task_gen():
    task = measurement_reduction_task_gen(
        PauliPartitionStrat.NonConflictingSets,
        GraphColourMethod.Lazy,
        CXConfigType.Snake,
    )
    assert task.n_in_wires == 1
    assert task.n_out_wires == 1
    #  set experiment up such that each expectation could be calculated by taking expectation
    #  over different bits of a single experiment
    c0 = (Circuit(3).X(0).X(1), 10, SymbolsDict())
    c1 = (Circuit(3).X(1).X(2), 10, SymbolsDict())
    qps_12_zz = QubitPauliString([Qubit(1), Qubit(2)], [Pauli.Z, Pauli.Z])
    qps_01_zz = QubitPauliString([Qubit(0), Qubit(1)], [Pauli.Z, Pauli.Z])
    qps_012_zzz = QubitPauliString(
        [Qubit(0), Qubit(1), Qubit(2)], [Pauli.Z, Pauli.Z, Pauli.Z]
    )
    t0 = ObservableTracker(QubitPauliOperator({qps_01_zz: 0.5, qps_12_zz: 1.0}))
    t1 = ObservableTracker(QubitPauliOperator({qps_012_zzz: 0.7, qps_01_zz: 0.8}))
    wire = [(c0, t0), (c1, t1)]

    # [0] for tuple
    output = task([wire])[0]
    assert len(output) == 2

    info00 = output[0][1]._qps_to_indices[qps_12_zz]
    info01 = output[0][1]._qps_to_indices[qps_01_zz]
    info10 = output[1][1]._qps_to_indices[qps_012_zzz]
    info11 = output[1][1]._qps_to_indices[qps_01_zz]
    # check each string only needs one measurement circuit
    assert len(info00) == 1
    assert len(info01) == 1
    assert len(info10) == 1
    assert len(info11) == 1
    # check each measurement circuit is at index 0
    assert info00[0][0] == 0
    assert info01[0][0] == 0
    assert info10[0][0] == 0
    assert info11[0][0] == 0
    # check bits are correct
    assert [Bit(1), Bit(2)] == info00[0][1]
    assert [Bit(0), Bit(1)] == info01[0][1]
    assert [Bit(0), Bit(1), Bit(2)] == info10[0][1]
    assert [Bit(0), Bit(1)] == info11[0][1]
    # check no result expects inversion of expectation
    assert info00[0][2] == False
    assert info01[0][2] == False
    assert info10[0][2] == False
    assert info11[0][2] == False


def test_gen_me_MitEx():
    # set up experiment with guaranteed expectations
    # confirm that despite measurement reduction reducing the number
    # of circuits simulated, the expectation value is still correct
    me = gen_MeasurementReduction_MitEx(AerBackend())

    c0 = (Circuit(3).X(0).X(1), 10, SymbolsDict())
    c1 = (Circuit(3).X(1).X(2), 10, SymbolsDict())
    qps_12 = QubitPauliString([Qubit(1), Qubit(2)], [Pauli.Z, Pauli.Z])
    qps_01 = QubitPauliString([Qubit(0), Qubit(1)], [Pauli.Z, Pauli.Z])
    qps_012 = QubitPauliString(
        [Qubit(0), Qubit(1), Qubit(2)], [Pauli.Z, Pauli.Z, Pauli.Z]
    )
    t0 = ObservableTracker(QubitPauliOperator({qps_01: 0.5, qps_12: 1.0}))
    t1 = ObservableTracker(QubitPauliOperator({qps_012: 0.7}))
    experiment = [(c0, t0), (c1, t1)]

    res = me.run(experiment)
    # correct expectations
    assert len(res) == 2
    assert res[0][qps_01] == 0.5
    assert res[0][qps_12] == -1.0
    assert res[1][qps_012] == 0.7


if __name__ == "__main__":
    test_measurement_reduction_task_gen()
    test_gen_me_MitEx()

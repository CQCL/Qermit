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


import pytest  # type: ignore

from qermit import (  # type: ignore
    SymbolsDict,
    MeasurementCircuit,
    ObservableTracker,
)

from pytket.circuit import Circuit, fresh_symbol, Qubit, Bit  # type: ignore
from pytket.pauli import QubitPauliString, Pauli  # type: ignore
from pytket.utils import QubitPauliOperator
from pytket.extensions.qiskit import AerBackend  # type: ignore

# tests for SymbolsDict class in utils
def test_SymbolsDict_circuit_constructor() -> None:
    test_circuit = Circuit(4)
    symbols = []
    for i in range(4):
        sym = fresh_symbol("a" + str(i))
        test_circuit.Ry(sym, i)
        symbols.append(sym)

    sd = SymbolsDict.symbols_from_circuit(test_circuit)
    for i in range(len(symbols)):
        sd.add_value(symbols[i], i)

    assert sd._symbolic_map[symbols[0]] == 0
    assert sd._symbolic_map[symbols[1]] == 1
    assert sd._symbolic_map[symbols[2]] == 2
    assert sd._symbolic_map[symbols[3]] == 3


def test_SymbolsDict_dict_constructor() -> None:
    symbols_dict = dict()
    for i in range(4):
        symbols_dict[fresh_symbol("a" + str(i))] = i
    sd = SymbolsDict.symbols_from_dict(symbols_dict)
    assert sd._symbolic_map == symbols_dict


def test_SymbolsDict_list_constructor() -> None:
    string_list = ["b{}".format(str(i)) for i in range(4)]
    sd = SymbolsDict.symbols_from_list(string_list)
    sd.add_symbol("b4")
    sd.set_values(range(5))
    assert sd._symbolic_map[fresh_symbol(string_list[0])] == 0
    assert sd._symbolic_map[fresh_symbol(string_list[1])] == 1
    assert sd._symbolic_map[fresh_symbol(string_list[2])] == 2
    assert sd._symbolic_map[fresh_symbol(string_list[3])] == 3
    assert sd._symbolic_map[fresh_symbol("b4")] == 4

    symb_b5 = fresh_symbol("b5")
    sd.add_symbol(symb_b5)
    sd.add_value(symb_b5, 5)
    assert sd._symbolic_map[symb_b5] == 5


# tests for MeasurementCircuit class in utils
def test_basic_MeasurementCircuit_constructor() -> None:
    test_circuit = Circuit(4)
    symbols = []
    for i in range(4):
        sym = fresh_symbol("c" + str(i))
        test_circuit.Ry(sym, i)
        symbols.append(sym)

    mc = MeasurementCircuit(test_circuit)
    sd = SymbolsDict.symbols_from_list(symbols)

    assert mc._symbols._symbolic_map[symbols[0]] == sd._symbolic_map[symbols[0]]
    assert mc._symbols._symbolic_map[symbols[1]] == sd._symbolic_map[symbols[1]]
    assert mc._symbols._symbolic_map[symbols[2]] == sd._symbolic_map[symbols[2]]
    assert mc._symbols._symbolic_map[symbols[3]] == sd._symbolic_map[symbols[3]]


def test_MeasurementCircuit_constructor() -> None:
    symbols_dict = dict()
    test_circuit = Circuit(4)
    comparison_circuit = Circuit(4)
    for i in range(4):
        sym = fresh_symbol("d" + str(i))
        test_circuit.Ry(sym, i)
        comparison_circuit.Ry(i, i)
        symbols_dict[sym] = i
    sd = SymbolsDict.symbols_from_dict(symbols_dict)
    mc = MeasurementCircuit(test_circuit, sd)

    assert mc.get_parametric_circuit() == comparison_circuit


# tests for ObservableTracker class in utils
def test_ObservableTracker_constructor() -> None:
    qps_0 = QubitPauliString(
        [Qubit(0), Qubit(1), Qubit(2)], [Pauli.X, Pauli.Y, Pauli.Z]
    )
    qps_1 = QubitPauliString(
        [Qubit(3), Qubit(1), Qubit(4)], [Pauli.Y, Pauli.Z, Pauli.Z]
    )
    operator = QubitPauliOperator(
        {
            qps_0: 1.0,
            qps_1: 1.0,
        }
    )
    test_ot = ObservableTracker(operator)
    assert test_ot._qubit_pauli_operator == operator
    assert len(test_ot._qps_to_indices[qps_0]) == 0
    assert len(test_ot._qps_to_indices[qps_1]) == 0
    assert len(test_ot._measurement_circuits) == 0
    assert len(test_ot._partitions) == 0


def test_ObservableTracker_measurement_circuits_methods() -> None:
    # make operator for example
    # make explicitly here for ease of parsing example
    qps_0 = QubitPauliString(
        [Qubit(0), Qubit(1), Qubit(2)], [Pauli.X, Pauli.Y, Pauli.Z]
    )
    qps_1 = QubitPauliString(
        [Qubit(3), Qubit(1), Qubit(4)], [Pauli.Y, Pauli.Z, Pauli.Z]
    )
    operator = QubitPauliOperator(
        {
            qps_0: 1.0,
            qps_1: 1.0,
        }
    )
    test_ot = ObservableTracker(operator)
    # make measurement circuits for addition
    circuit_0 = Circuit(5)
    circuit_1 = Circuit(5)
    symbols_dict_0 = dict()
    symbols_dict_1 = dict()

    for i in range(5):
        # new symbol
        sym_0 = fresh_symbol("mc0_" + str(i))
        sym_1 = fresh_symbol("mc1_" + str(i))
        # add basic symbolic gates to both measurement circuits
        circuit_0.Ry(sym_0, i)
        circuit_0.add_bit(Bit(i))
        circuit_1.Ry(sym_1, i)
        circuit_1.add_bit(Bit(i))
        # update symbols dict
        symbols_dict_0[sym_0] = i
        symbols_dict_1[sym_1] = i

    circuit_0.H(Qubit(0))
    circuit_0.Rx(0.5, Qubit(1))
    circuit_1.Rx(0.5, Qubit(0))
    for i in range(5):
        circuit_0.Measure(Qubit(i), Bit(i))
        circuit_1.Measure(Qubit(i), Bit(i))

    mc_0 = MeasurementCircuit(circuit_0, SymbolsDict.symbols_from_dict(symbols_dict_0))
    mc_1 = MeasurementCircuit(circuit_1, SymbolsDict.symbols_from_dict(symbols_dict_1))

    measurement_info_0 = (qps_0, [Bit(0), Bit(1), Bit(2)], False)
    measurement_info_1 = (qps_1, [Bit(3), Bit(1), Bit(4)], False)

    test_ot.add_measurement_circuit(mc_0, [measurement_info_0])
    test_ot.add_measurement_circuit(mc_1, [measurement_info_1, measurement_info_0])

    measurement_circuits_0 = test_ot.get_measurement_circuits(qps_0)
    measurement_circuits_1 = test_ot.get_measurement_circuits(qps_1)

    assert measurement_circuits_0[0] == mc_0
    assert measurement_circuits_0[1] == mc_1
    assert measurement_circuits_1[0] == mc_1

    all_circuits = [mc.get_parametric_circuit() for mc in test_ot.measurement_circuits]
    # noiseless simulator
    backend = AerBackend()
    handles = backend.process_circuits(all_circuits, 10)
    results = backend.get_results(handles)
    final_qpo = test_ot.get_expectations(results)

    assert final_qpo[qps_1] == 1

    with pytest.raises(ValueError):
        test_ot.get_expectations([results[0]])


def test_ObservableTracker_add_operator() -> None:
    string_0 = QubitPauliString(
        [Qubit(0), Qubit(1), Qubit(2)], [Pauli.X, Pauli.Y, Pauli.Z]
    )
    string_1 = QubitPauliString(
        [Qubit(3), Qubit(1), Qubit(4)], [Pauli.Y, Pauli.Z, Pauli.Z]
    )
    qpo_0 = QubitPauliOperator({string_0: 0.6})
    qpo_1 = QubitPauliOperator({string_1: 0.5})
    base_ot = ObservableTracker(qpo_0)
    base_ot.extend_operator(qpo_1)
    combined_operator = QubitPauliOperator({string_0: 0.6, string_1: 0.5})
    assert base_ot._qubit_pauli_operator == combined_operator


def test_ObservableTracker_get_empty_strings() -> None:
    qps_0 = QubitPauliString(
        [Qubit(0), Qubit(1), Qubit(2)], [Pauli.X, Pauli.Y, Pauli.Z]
    )
    qps_1 = QubitPauliString(
        [Qubit(3), Qubit(1), Qubit(4)], [Pauli.Y, Pauli.Z, Pauli.Z]
    )
    operator = QubitPauliOperator(
        {
            qps_0: 1.0,
            qps_1: 1.0,
        }
    )
    test_ot = ObservableTracker(operator)
    strings = list(operator._dict.keys())
    assert test_ot.get_empty_strings() == strings


def test_ObservableTracker_copy() -> None:
    qps_0 = QubitPauliString(
        [Qubit(0), Qubit(1), Qubit(2)], [Pauli.X, Pauli.Y, Pauli.Z]
    )
    qps_1 = QubitPauliString(
        [Qubit(3), Qubit(1), Qubit(4)], [Pauli.Y, Pauli.Z, Pauli.Z]
    )
    operator = QubitPauliOperator(
        {
            qps_0: 1.0,
            qps_1: 1.0,
        }
    )
    base_ot = ObservableTracker(operator)

    copied_ot = ObservableTracker.from_ObservableTracker(base_ot)
    assert base_ot._qubit_pauli_operator == copied_ot._qubit_pauli_operator
    assert base_ot._qps_to_indices == copied_ot._qps_to_indices
    assert base_ot._measurement_circuits == copied_ot._measurement_circuits
    assert base_ot._partitions == base_ot._partitions


if __name__ == "__main__":
    test_SymbolsDict_circuit_constructor()
    test_SymbolsDict_dict_constructor()
    test_SymbolsDict_list_constructor()
    test_basic_MeasurementCircuit_constructor()
    test_MeasurementCircuit_constructor()
    test_ObservableTracker_constructor()
    test_ObservableTracker_copy()
    test_ObservableTracker_get_empty_strings()
    test_ObservableTracker_measurement_circuits_methods()
    test_ObservableTracker_add_operator()

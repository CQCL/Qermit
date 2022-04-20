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


from pytket.circuit import Circuit, Bit  # type: ignore
from pytket.utils import (
    QubitPauliOperator,
    expectation_from_counts,
)
from pytket.pauli import QubitPauliString  # type: ignore
from pytket.backends.backendresult import BackendResult
from copy import copy
from sympy import Symbol  # type: ignore
from typing import Iterable, Dict, Union, Tuple, List
from collections import OrderedDict
from numpy import ndarray


class SymbolsDict(object):
    """
    A helper class for standardising interfacing with Circuit Symbolics in qermit.
    Methods take different containers that hold some kind of symbols representation
    and return a SymbolsDict object.
    Methods access self._symbolic_map or use other accessors to modify or add new
    symbols.
    """

    def __init__(self):
        """
        Default constructor, creates an empty OrderedDict() object for future symbols to be added to.
        """
        self._symbolic_map: Dict[Symbol, Union[None, float]] = OrderedDict()

    @classmethod
    def symbols_from_circuit(cls, circuit: Circuit) -> "SymbolsDict":
        """
        Given a pytket Circuit, returns a SymbolsDict object capturing
        given circuits free symbols.

        :param circuit: Pytket circuit with potential symbols.
        :type circuit: Circuit

        """
        mit_symbols = cls()
        for sym in circuit.free_symbols():
            mit_symbols.add_symbol(sym)
        return mit_symbols

    @classmethod
    def symbols_from_dict(
        cls, symbol_dict: Dict[Symbol, Union[None, float]]
    ) -> "SymbolsDict":
        """
        Assigns to mit_symbols attribute _symbolic_map straight from passed dictionary
        of Symbol to None/float.

        :param symbol_dict: Dictionary from Circuit symbolics to values.
        :type symbol_dict: Dict[Symbol, Union[None, float]

        """
        mit_symbols = cls()
        mit_symbols._symbolic_map = symbol_dict
        return mit_symbols

    @classmethod
    def symbols_from_list(
        cls, symbols_list: Iterable[Union[Symbol, str]]
    ) -> "SymbolsDict":
        """
        Adds all symbols (or string representing Symbol) as dict entries with no value.

        :param symbols_list: A list of strings representing Symbols or Symbols.
        :type symbols_list: Iterable[Union[Symbol, str]

        """
        mit_symbols = cls()
        for sym in symbols_list:
            mit_symbols.add_symbol(sym)
        return mit_symbols

    @property
    def symbols_list(self) -> Iterable[Symbol]:
        """
        Returns all symbols held in dictionary of symbols in SymbolsDict object.

        :return: Iterable containing all keys from _symbolic_map, i.e. all Symbols
        :rtype: Iterable[Symbol]
        """
        for s in self._symbolic_map.keys():
            yield s

    def add_symbol(self, symbol: Union[str, Symbol]):
        """
        Adds any passed Symbol (in string form or sympy Symbol type) as a key to dictionary with None value assigned.

        :param symbol: Symbol to be added to self._symbolic_map
        :type symbol: Union[str, Symbol]
        """
        if isinstance(symbol, str):
            sym = Symbol(symbol)
            self._symbolic_map[sym] = None
        elif isinstance(symbol, Symbol):
            self._symbolic_map[symbol] = None
        else:
            msg = f"""
            Argument symbol is of invalid type: {type(symbol)}.
            """
            raise TypeError(msg)

    def get_symbolic_map(self, symbol_values: ndarray) -> Dict[Symbol, float]:
        """
        Assigns given values in parameters to keys in self._symbolic_map in order, for a new
        dictionary object. Returns just this dictionary type.

        :param symbol_values: Ordered values to match to ordered keys for new dict object.
        :type symbol_values: ndarray

        :return: New dict object mapping symbol to value.
        :rtype: Dict[Symbol, float]
        """
        _map = {}
        for symbol, value in zip(self._symbolic_map.keys(), symbol_values):
            _map[symbol] = value
        return _map

    def set_values(self, symbol_values: ndarray):
        """
        Assigns given values in parameters to keys in self._symbolic_map in order, for a new
        dictionary object.

        :param symbol_values: Orderd values to match to ordered keys for new dict object.
        :type symbol_values: ndarray

        :return: New dict object mapping symbol to value.
        :rtype: Dict[Symbol, float]
        """
        _map = {}
        for symbol, value in zip(self._symbolic_map.keys(), symbol_values):
            _map[symbol] = value
        self._symbolic_map = _map

    def add_value(self, symbol: Symbol, value: float):
        """
        Assigns value to self._symbolic_map[symbol]. If symbol not in object then throws an error.

        :param symbol: Symbol to have value assigned.
        :type symbol: Symbol
        :param value: Value to assign to symbol.
        :type value: float
        """
        if symbol in self._symbolic_map:
            self._symbolic_map[symbol] = value
        else:
            raise ValueError("Symbol {} not in object.".format(symbol))

    def __str__(self):
        return f"<SymbolsDict::{len(self._symbolic_map)}>"

    def __repr__(self):
        return str(self)


class MeasurementCircuit(object):
    """
    Stores a single measurement circuit that captures one or multiple observable estimations
    for some Ansatz Circuit.
    """

    def __init__(self, symbolic_circuit: Circuit, symbols: SymbolsDict = None):
        """
        Stores information required to instantiate any MeasurementCircuit with parameterised symbols.

        :param symbolic_circuit: Measurement circuit, may or may not have symbolics.
        :type symbolic_circuit: Circuit
        :param symbols: SymbolsDict object holding symbols and values for all symbols in Circuit. Default none if circuit not symbolic.
        :type symbols: SymbolsDict
        """
        self._symbolic_circuit: Circuit = symbolic_circuit
        if symbols is None:
            self._symbols: SymbolsDict = SymbolsDict.symbols_from_circuit(
                symbolic_circuit
            )
        elif isinstance(symbols, SymbolsDict):
            self._symbols = symbols
        else:
            raise ValueError("Passed symbols object of incorrect type.")

    @property
    def circuit(self) -> Circuit:
        """
        Returns measurement circuit stored in oracle.

        :return: Circuit in oracle
        :rtype: Circuit
        """
        return self._symbolic_circuit

    @property
    def symbols(self) -> Tuple[Symbol, ...]:
        """
        Converts symbols_list property held in SymbolsDict to a tuple and returns it.

        :return: All Symbols in object
        :rtype: Tuple[List[Symbol]]
        """
        return tuple(self._symbols.symbols_list)

    def get_parametric_circuit(self) -> Circuit:
        """
        Substitutes parameters held in SymbolDict into copy of circuit and returns.

        :return: Substituted circuit
        :rtype: Circuit
        """
        _circuit = self._symbolic_circuit.copy()
        _circuit.symbol_substitution(self._symbols._symbolic_map)
        return _circuit


MeasurementInfo = Tuple[QubitPauliString, List[Bit], bool]


class ObservableTracker:
    """
    Stores all measurement circuits required to get observable expectations for each
    QubitPauliString in a given QubitPauliOperator.
    """

    def __init__(self, qubit_pauli_operator: QubitPauliOperator = QubitPauliOperator()):
        """
        Default constructor, creates an empty dict object for mapping QubitPauliStrings to measurement circuits
        and the qubits measured to get expectation, along with an empty list for storing measurement circuits and
        a list for storing partitions.

        :param qubit_pauli_operator: QubitPauliOperator for which given ObservableTracker is expected
            to retain measurement circuits all QubitPauliString keys for before any Backend execution.
        :type qubit_pauli_strings: List[QubitPauliString]
        """
        self._qubit_pauli_operator = qubit_pauli_operator
        # indices being index in measurement circuits
        self._qps_to_indices: Dict[
            QubitPauliString, List[Tuple[int, List[Bit], bool]]
        ] = dict()
        for k in self._qubit_pauli_operator._dict.keys():
            self._qps_to_indices[k] = list()
        self._measurement_circuits: List[MeasurementCircuit] = list()
        self._partitions: List[QubitPauliString] = list()

    def from_ObservableTracker(to_copy: "ObservableTracker") -> "ObservableTracker":
        """
        Copies each class attribute from to_copy to self. Returns self.

        :param to_copy: An alternative ObservableTracker for making a copy of.
        :type to_copy: 'ObservableTracker'

        :return: New ObservableTracker object
        """
        # these variables could be mutated in the first ObservableTracker and effect
        # this one
        # To fix, copy everything
        new_obj = ObservableTracker(copy(to_copy._qubit_pauli_operator))

        new_obj._qps_to_indices = copy(to_copy._qps_to_indices)
        new_obj._measurement_circuits = copy(to_copy._measurement_circuits)
        new_obj._partitions = copy(to_copy._partitions)
        return new_obj

    def __str__(self):
        return (
            f"<ObservableTracker::{len(self._measurement_circuits)}MeasurementCircuits>"
        )

    def __repr__(self):
        return str(self)

    def clear(self):
        """
        Erases all held information that is not the qubit pauli operator.
        """
        self._qps_to_indices = dict()
        for k in self._qubit_pauli_operator._dict.keys():
            self._qps_to_indices[k] = list()
        self._measurement_circuits: List[MeasurementCircuit] = list()
        self._partitions: List[QubitPauliString] = list()

    def modify_coefficients(
        self, new_coefficients: List[Tuple[QubitPauliString, float]]
    ):
        """
        Updates coefficients in held QubitPauliOperator with new coefficients. Each QubitPauliString
        must already be in self._qubit_pauli_operator

        :param new_coefficients: Each Tuple contains a QubitPauliString a new coefficient.
        :type new_coefficients: List[Tuple[QubitPauliString, float]]
        """
        for coeff in new_coefficients:
            if coeff[0] not in self._qubit_pauli_operator._dict:
                raise ValueError(
                    "Given string {} not held in ObservableTracker object.".format(
                        coeff[0]
                    )
                )
            self._qubit_pauli_operator._dict[coeff[0]] = coeff[1]

    def extend_operator(self, new_operator: QubitPauliOperator):
        """
        Extends self._qubit_pauli_operator to include tuples in passed operator.

        :param new_operator: Each QubitPauliString and coefficient added to held operator.
        :type new_operator: QubitPauliOperator
        """
        self._qubit_pauli_operator += new_operator

    def remove_strings(self, strings: List[QubitPauliString]):
        """
        Removes passed qubit pauli strings from held QubitPauliOperator and dict from string to index.

        :param strings: Qubit Pauli Strings no longer required to be measured by ObservableTracker
        :type strings: List[QubitPauliString]
        """
        for qps in strings:
            self._qps_to_indices.pop(qps, None)
            self._qubit_pauli_operator._dict.pop(qps, None)

    @property
    def qubit_pauli_operator(self):
        """
        Returns stored qubit pauli operator

        :return: QubitPauliOperator object stored in class
        :rtype: QubitPauliOperator
        """
        return self._qubit_pauli_operator

    def add_measurement_circuit(
        self, circuit: MeasurementCircuit, measurement_info: List[MeasurementInfo]
    ):
        """
        Adds given measurement circuit to stored _measurement_circuits attribute and for each qubit pauli string and qubits in associated
        strings, updates dictionary between string and its measurement circuit + bit to measure and whether result should be inverted.

        :param circuit: Measurement circuit to run to get results.
        :type circuit: MeasurementCircuit
        :param measurement_info: Each entry contains a QubitPauliString, the bits required to take expectation over in resulting result
            and a bool signifying whether expectation should be inverted when taking result.
        :type measurement_info: List[MeasurementInfo] i.e. List[Tuple[QubitPauliString, List[Bit], bool]]
        """
        self._measurement_circuits.append(circuit)
        index = len(self._measurement_circuits) - 1

        # assume that if strings have the same circuit, they must commute
        self._partitions.append([m[0] for m in measurement_info])
        for s in measurement_info:
            string = s[0]
            bits = s[1]
            invert = s[2]
            if s[0] not in self._qps_to_indices:
                raise ValueError(
                    "ObservableTracker object does not track {}.".format(s[0])
                )
            self._qps_to_indices[string].append((index, bits, invert))

    def get_measurement_circuits(
        self, string: QubitPauliString
    ) -> List[MeasurementCircuit]:
        """
        Returns the measurements required to be run for a single QubitPauliString's expectation.

        :param string: QubitPauliString of interest.
        :type string: QubitPauliString

        :return: Measurement Circuit run to find expection of QubitPauliString for some undefined ansatz circuit.
        :rtype: MeasurementCircuit
        """
        indices = [t[0] for t in self._qps_to_indices[string]]
        circuits = [self._measurement_circuits[i] for i in indices]
        return circuits

    def check_string(self, string: QubitPauliString) -> bool:
        """
        Returns true if given QubitPauliString has a measurement circuit stored in self._measurement_circuits.

        :param string: Operator measurement circuit existence being checked for.
        :type string: QubitPauliString

        :return: True if string has measurement circuit, false if not.
        :rtype: bool
        """
        if string not in self._qps_to_indices:
            return False
        if len(self._qps_to_indices[string]) > 0:
            return True
        else:
            return False

    def get_empty_strings(self) -> List[QubitPauliString]:
        """
        Returns all strings in operator that don't have some assigned MeasurementCircuit.

        :return: Strings that require some MeasurementCircuit to be set
        :rtype: List[QubitPauliString]
        """
        output = []
        for string in self._qubit_pauli_operator._dict:
            if not self.check_string(string):
                output.append(string)
        return output

    @property
    def measurement_circuits(self) -> List[MeasurementCircuit]:
        """
        Returns all measurement circuits aded to ObservableTracker via get_measurement_circuit.

        :return: All measurement circuits held in ObservableTracker self._measurement_circuits attirbute.
        :rtype: List[MeasurementCircuit]
        """
        return self._measurement_circuits

    def get_expectations(self, results: List[BackendResult]) -> QubitPauliOperator:
        """
        For given list of results, returns a QubitPauliOperator giving an expectation for each QubitPauliString
        held in self._qps_to_indices. Expectation derived by taking parity of counts.

        :param results: Result objects to derive counts and then an expectation from.
        :type result: BackendResult

        :return: Expectation for each QubitPauliString in self._qps_to_indices
        :type: QubitPauliOperator
        """
        max_index = len(results) - 1
        results_dict = dict()
        # find expectation for each qubit pauli string stored in dict
        for qps in self._qps_to_indices:
            expectation = 0
            # measure info stores result index of interest, bits of choice and
            # whether result should be inverted
            for measure_info in self._qps_to_indices[qps]:
                result_index = measure_info[0]
                # suggests something has gone wrong with MitEx piping of tasks
                if result_index > max_index:
                    raise ValueError(
                        "Desired index {} greater than max index {} of results.".format(
                            result_index, max_index
                        )
                    )

                result = results[result_index]
                bits = measure_info[1]
                invert = measure_info[2]
                counts = result.get_counts(bits)
                expectation += ((-1) ** invert) * expectation_from_counts(counts)
            # once expectation has been derived from all suitable results, add to dictionary
            coeff = self._qubit_pauli_operator[qps]
            results_dict[qps] = expectation * coeff
        # package in QubitPauliOperator as MitEx uses
        return QubitPauliOperator(results_dict)

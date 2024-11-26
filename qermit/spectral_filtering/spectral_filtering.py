from copy import copy, deepcopy
from itertools import product
from typing import Any, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray
from pytket.backends import Backend
from pytket.pauli import QubitPauliString
from pytket.utils import QubitPauliOperator
from scipy import fft, interpolate  # type: ignore

from qermit import AnsatzCircuit, ObservableExperiment, SymbolsDict
from qermit.taskgraph.mitex import MitEx, gen_compiled_MitRes
from qermit.taskgraph.mittask import MitTask
from qermit.taskgraph.task_graph import TaskGraph

from .signal_filter import SignalFilter, SmallCoefficientSignalFilter

# TODO: The type annotation for NDArrays should be improved when newer
# versions of python are supported. Reference the documentation
# for the 'true' typing.


def gen_result_extraction_task() -> MitTask:
    """Generates task which extracts a result at coordinates specified
    by symbol values from grid of results.

    :return: Task interpolating results.
    """

    def task(
        obj,
        result_list: List[Dict[QubitPauliString, NDArray]],
        obs_exp_list: List[ObservableExperiment],
        points_list: List[List[float]],
    ) -> Tuple[List[QubitPauliOperator]]:
        """Task interpolating result at point specified by symbol values
        in the circuits of obs_exp_list. Elements of `result_list` are used
        as discrete grid for interpolation. `points_list` are the axis
        of the grid.

        :param result_list: List of dictionaries or results grids. Entries in
            result_list correspond to experiments in obs_exp_list. Keys of
            each entry correspond to the pauli strings in each observable
            of the corresponding experiment. Values of each entry are grids
            with axis defined by points_list. Each point on the grid
            corresponds to the circuit in obs_exp_list with symbols substituted
            for the grid point value.
        :param obs_exp_list: List of observable experiments. The value of
            the symbols in these circuits are used as the points to
            interpolate to. Note that the QubitPauliString which comprise
            the measurement QubitPauliOperator in each ObservableExperiment
            should match those in the corresponding dictionary of
            `result_list`.
        :param points_list: List of values taken by each symbol on the
            result grid.
        :return: List of interpolated results.
        """

        interpolated_result_list = []
        for result, points, obs_exp in zip(result_list, points_list, obs_exp_list):
            # Extract point to be interpolated to. This is the symbol values
            # given in the initial experiment definition.
            interpolation_point = list(
                obs_exp.AnsatzCircuit.SymbolsDict._symbolic_map.values()
            )

            # For each QubitPauliString in the experiment QubitPauliOperator,
            # interpolate it's value at the given point.
            interpolated_qpo = deepcopy(obs_exp.ObservableTracker.qubit_pauli_operator)

            if set(interpolated_qpo._dict.keys()) != set(result.keys()):
                raise Exception(
                    "The QubitPauliStrings in `obs_exp_list`, \
                        and those in the `result_list` do not match."
                )

            for qps in interpolated_qpo._dict.keys():
                interpolated_qpo._dict[qps] = interpolate.interpn(
                    points, result[qps], interpolation_point
                )[0]
            interpolated_result_list.append(interpolated_qpo)

        return (interpolated_result_list,)

    return MitTask(
        _label="ResultExtraction", _n_out_wires=1, _n_in_wires=3, _method=task
    )


def gen_mitigation_task(signal_filter: SignalFilter) -> MitTask:
    """Generates task acting `signal_filter` on the fourier coefficients
    of a grid of results.

    :param signal_filter: Method of filtering the signal.
    :return: Task performing filtering.
    """

    def task(
        obj, result_grid_list: List[Dict[QubitPauliString, NDArray]]
    ) -> Tuple[List[Dict[QubitPauliString, NDArray]]]:
        """Task acting `signal_filter` on the value of the dictionaries in
        `result_grid_list`.

        :param result_grid_list: List of dictionaries mapping
            QubitPauliString to arrays.
        :return: Grids having had `signal_filter` applied.
        """

        mitigated_result_grid_list = []
        for result_grid in result_grid_list:
            mitigated_result_grid_dict = dict()
            for qps, grid in result_grid.items():
                mitigated_result_grid_dict[qps] = signal_filter.filter(grid)
            mitigated_result_grid_list.append(mitigated_result_grid_dict)

        return (mitigated_result_grid_list,)

    return MitTask(
        _label="SignalFilterMitigation",
        _n_out_wires=1,
        _n_in_wires=1,
        _method=task,
    )


def gen_fft_task() -> MitTask:
    """Generates task which performs Fast Fourier Transform
    of grids of values.

    :return: Task performing Fast Fourier Transform
    """

    def task(
        obj, result_grid_dict_list: List[Dict[QubitPauliString, NDArray]]
    ) -> Tuple[List[Dict[QubitPauliString, NDArray]]]:
        """Task performing Fast Fourier Transform on each value of
        the dictionaries in the list `result_grid_dict_list`. This uses fft,
        which is a scipy method.

        :param result_grid_dict_list: List of dictionaries mapping
            QubitPauliString to arrays of results.
        :return: List of dictionaries mapping
            QubitPauliString to the Fast Fourier Transform of the values
            in the dictionaries of `result_grid_dict_list`.
        """

        fft_result_grid_list = []
        for result_grid_dict in result_grid_dict_list:
            # Perform the FFT on grids corresponding to each QubitPauliString.
            fft_result_grid_dict = dict()
            for qps, exp_val_grid in result_grid_dict.items():
                fft_result_grid_dict[qps] = fft.fftn(exp_val_grid)
            fft_result_grid_list.append(fft_result_grid_dict)

        return (fft_result_grid_list,)

    return MitTask(_label="FFT", _n_out_wires=1, _n_in_wires=1, _method=task)


def gen_ndarray_to_dict_task() -> MitTask:
    """Generates task reshaping an array of QubitPauliOperator into a
    dictionary with QubitPauliStrings as keys and an array of the appropriate
    coefficients as values.

    :return: Task reshaping QubitPauliOperator into a
        dictionary with QubitPauliStrings as keys and an array of
        the appropriate coefficients as values.
    """

    def task(
        obj, result_grid_list: List[NDArray]
    ) -> Tuple[List[Dict[QubitPauliString, NDArray]]]:
        """Task reshaping an arrays of QubitPauliOperator in the list
        `result_grid_list` into dictionaries with QubitPauliStrings as keys
        and an array of the appropriate coefficients as values.

        :param result_grid_list: List of QubitPauliOperators to be reshaped.
        :return: List of QubitPauliOperators reshaped as dictionaries from
            QubitPauliStrings to arrays of coefficients.
        """

        result_dict_list = []
        for qpo_result_grid in result_grid_list:
            # Take the QubitPauliOperator that is being measured from the 0
            # coordinate element of the grid.
            zero_qpo_result_grid = qpo_result_grid[
                tuple(0 for _ in qpo_result_grid.shape)
            ]
            # For each QubitPauliString in the QubitPauliOperator, add a
            # key and initialise an empty grid. The new grid will contain
            # the expectations of the individual QubitPauliStrings.
            result_grid_dict = dict()
            for key in zero_qpo_result_grid._dict.keys():
                result_grid_dict[key] = np.empty(qpo_result_grid.shape, dtype=float)

            # For each point on the grid, extract the expections of the
            # given QubitPauliString
            grid_point_val_list = [
                [i for i in range(size)] for size in qpo_result_grid.shape
            ]
            for grid_point in product(*grid_point_val_list):
                qpo_result_dict = qpo_result_grid[grid_point]._dict
                for qps, exp_val in qpo_result_dict.items():
                    result_grid_dict[qps][grid_point] = exp_val

            result_dict_list.append(result_grid_dict)

        return (result_dict_list,)

    return MitTask(
        _label="NDArrayToDict",
        _n_out_wires=1,
        _n_in_wires=1,
        _method=task,
    )


def gen_inv_fft_task() -> MitTask:
    """Generates task performing the inverse Fast Fourier Transform
    on each value in the list of dictionaries.

    :return: Task performing Fast Fourier Transform.
    """

    def task(
        obj, result_grid_list: List[Dict[QubitPauliString, NDArray]]
    ) -> Tuple[List[Dict[QubitPauliString, NDArray]]]:
        """Task performing the inverse Fast Fourier Transform on each
        value in the list of dictionaries `result_grid_list`.
        The dictionary keys are unchanged by this task. This uses fft,
        which is a scipy method.

        :param result_grid_list: List of dictionaries, where values are
            to have the inverse Fast Fourier Transform performed on them.
        :return: List of dictionaries, where all values are the
            Fast Fourier Transform of the values of `result_grid_list`
        """

        # Iterate through results and invert FFT
        ifft_result_grid_list = []
        for result_grid_dict in result_grid_list:
            ifft_result_grid_dict = dict()
            for qps, grid in result_grid_dict.items():
                ifft_result_grid_dict[qps] = fft.ifftn(grid)
            ifft_result_grid_list.append(ifft_result_grid_dict)

        return (ifft_result_grid_list,)

    return MitTask(
        _label="InvFFT",
        _n_out_wires=1,
        _n_in_wires=1,
        _method=task,
    )


def gen_flatten_task() -> MitTask:
    """Generates task which transforms a list of ndarrays of
    ObservableExperiments into a list of ObservableExperiments. These
    ObservableExperiments can then be run in sequence.

    :return: Generates task which transforms a list of ndarrays of
        ObservableExperiments into a list of ObservableExperiments
    """

    def task(
        obj, grid_list: List[NDArray]
    ) -> Tuple[List[ObservableExperiment], List[int], List[Tuple[int, ...]]]:
        """Task which transforms a list of ndarrays of
        ObservableExperiments into a list of ObservableExperiments

        :param grid_list: List of ndarrays of ObservableExperiments
        :return: List of ObservableExperiments, and details about the
            original and new structure of the list. In particular
            `length_list` is the length of the sublist of the new list
            which contains each ndarray. `shape_list` contains the
            shape of each ndarray.
        """

        # Store structure of flattened grid as list of dictionaries.
        shape_list = []
        length_list = []
        # ObservableExperiments, currently stored in a grid,
        # are flattened to a single list.
        flattened_grid_list = []
        for grid in grid_list:
            flattened_grid = grid.flatten()
            length_list.append(len(flattened_grid))
            flattened_grid_list += list(flattened_grid)
            shape_list.append(grid.shape)

        return (
            flattened_grid_list,
            length_list,
            shape_list,
        )

    return MitTask(
        _label="Flatten",
        _n_out_wires=3,
        _n_in_wires=1,
        _method=task,
    )


def gen_reshape_task() -> MitTask:
    """Generates task which reshapes a list of QubitPauliOperator into
    a list of ndarrays of QubitPauliOperator. This can be used to have the
    effect for reversing the task generated by `gen_flatten_task`.

    :return: Task which reshapes a list of QubitPauliOperator into
        a list of ndarrays of QubitPauliOperator.
    """

    def task(
        obj,
        result_list: List[QubitPauliOperator],
        length_list: List[int],
        shape_list: List[Tuple[int]],
    ) -> Tuple[List[NDArray]]:
        """Task which reshapes a list of QubitPauliOperator into
        a list of ndarrays of QubitPauliOperator.

        :param result_list: List of QubitPauliOperator to be reshaped.
        :param length_list: The length of the sublists of `result_list` which
            correspond to each ndarray.
        :param shape_list: The shape of the ndarray which each sublist
            should be reshaped into
        :return: The reshaped list of ndarrays.
        """

        position_list = [
            sum(sub_list)
            for sub_list in [length_list[:i] for i in range(len(length_list) + 1)]
        ]
        flattened_result_grid_list = [
            result_list[start:end]
            for start, end in zip(position_list[:-1], position_list[1:])
        ]
        result_grid_list = [
            np.reshape(np.array(flattened_result_grid), shape)
            for flattened_result_grid, shape in zip(
                flattened_result_grid_list, shape_list
            )
        ]

        return (result_grid_list,)

    return MitTask(
        _label="Reshape",
        _n_out_wires=1,
        _n_in_wires=3,
        _method=task,
    )


def gen_obs_exp_grid_gen_task() -> MitTask:
    """Generates task creating an ObservableExperiment for each point in the
    input meshgrid.

    :return: Task generating grid of experiments.
    """

    def task(
        obj,
        obs_exp_list: List[ObservableExperiment],
        obs_exp_sym_val_grid_list: List[NDArray],
    ) -> Tuple[List[NDArray]]:
        """Task generating a grid of ObservableExperiments.
        Each point in the grid corresponds to substituting the symbols in
        the circuit of each `ObservableExperiment` for a value in the
        meshgrids of `obs_exp_sym_val_grid_list`.

        :param obs_exp_list: A list of `ObservableExperiments`. This is
            this list of experiments to be mitigated, as requested by
            the user.
        :param obs_exp_sym_val_grid_list: A list of collections of meshgrids.
            Each symbol in the Circuit of each `ObservableExperiment` is
            represented in the meshgrids of `obs_exp_sym_val_grid_list`.
        :return: A grid of `ObservableExperiments`. Each `ObservableExperiment`
            includes a `SymbolDict` evaluated at a point in the grid of
            `obs_exp_sym_val_grid_list`.
        """

        # A grid of ObservableExperiment is generated for each
        # ObservableExperiment in obs_exp_list
        obs_exp_grid_list = []
        for obs_exp, sym_val_grid_list in zip(obs_exp_list, obs_exp_sym_val_grid_list):
            # Initialise empty grid of ObservableExperiment
            obs_exp_grid = np.empty(
                sym_val_grid_list[0].shape, dtype=ObservableExperiment
            )

            # Generate an ObservableExperiment for every symbol
            # value in the grid
            grid_point_val_list = [
                [i for i in range(size)] for size in sym_val_grid_list[0].shape
            ]
            for grid_point in product(*grid_point_val_list):
                # Generate dictionary mapping every symbol to it's value at
                # the given point in the grid.
                sym_map = {
                    sym: sym_val_grid[grid_point]
                    for sym_val_grid, sym in zip(
                        sym_val_grid_list,
                        obs_exp.AnsatzCircuit.SymbolsDict.symbols_list,
                    )
                }
                sym_dict = SymbolsDict().symbols_from_dict(sym_map)
                circ = obs_exp.AnsatzCircuit.Circuit.copy()
                anz_circ = AnsatzCircuit(circ, obs_exp.AnsatzCircuit.Shots, sym_dict)

                obs_exp_grid[grid_point] = ObservableExperiment(
                    anz_circ, obs_exp.ObservableTracker
                )

            obs_exp_grid_list.append(obs_exp_grid)

        return (obs_exp_grid_list,)

    return MitTask(
        _label="ObsExpGridGen",
        _n_out_wires=1,
        _n_in_wires=2,
        _method=task,
    )


def gen_symbol_val_gen_task(n_sym_vals: int) -> MitTask:
    """Generates task which produces a grid of values. The values are
    generated uniformly in the interval [0,2]
    (factors of pi give full coverage of the Bloch sphere) for each symbol.
    The points on the grid are equally spaces, and the number of values
    is the same in each dimension. In the context of the spectral filtering
    MitEx, the given circuit will be evaluated at each point in the grid.
    This is to say that the symbols in the circuit will be substituted for
    the values at each point in the grid.

    :param n_sym_vals: The number of values to be taken by each symbol
        in the circuit.
    :return: Task generating grid of symbol values.
    """

    def task(
        obj, obs_exp_list: List[ObservableExperiment]
    ) -> Tuple[List[ObservableExperiment], List[List[NDArray]]]:
        """Produces a grid of values taken by the symbols in
        the circuit. The values are generated uniformly in the interval [0,2]
        (factors of pi give full coverage) for each symbol. The points on the
        grid are equally spaces, and the number of values is the same in
        each dimension.

        :param obs_exp_list: List of ObservableExperiments to be run.
        :return: List of observable experiments is unchanged. For each
            experiment, a grid is generated.
        """

        # A collection of symbol value lists is generated for each
        # ObservableExperiment in obs_exp_list. Within each collection of
        # lists, there is a list for each of the symbols.
        return (
            obs_exp_list,
            [
                [
                    np.linspace(0, 2, n_sym_vals, endpoint=False)
                    for _ in obs_exp.AnsatzCircuit.SymbolsDict.symbols_list
                ]
                for obs_exp in obs_exp_list
            ],
        )

    return MitTask(
        _label="SymbolValGen",
        _n_out_wires=2,
        _n_in_wires=1,
        _method=task,
    )


def gen_wire_copy_task(n_in_wires: int, n_wire_copies: int) -> MitTask:
    """Generates task which copies each of the input wires `n_wire_copies`
    times. The output wires are repeated in the same order as the input wires.
    This is to say that input wires 1,2,...n will be outputted as
    1,2,...,n,...,1,2,...n.

    :param n_in_wires: The number of wires input into this task.
    :param n_wire_copies: The number of times the wires are to be copied.
    :return: Task which copies input wires.
    """

    def task(obj, *in_wires: Tuple[Any, ...]):
        """Task copying `in_wires`. The output wires are repeated in the
        same order as the input wires. This is to say that input wires
        1,2,...n will be outputted as 1,2,...,n,...,1,2,...n.

        :return: Copied wires
        """
        out_wires: Tuple = tuple()
        for _ in range(n_wire_copies):
            for wire in in_wires:
                out_wires = (*out_wires, copy(wire))
        return out_wires

    return MitTask(
        _label="WireCopy",
        _n_out_wires=n_in_wires * n_wire_copies,
        _n_in_wires=n_in_wires,
        _method=task,
    )


def gen_param_grid_gen_task() -> MitTask:
    """Generates task converting lists of symbol values into a meshgrid.

    :return: Task converting list of symbol values into a meshgrid.
    """

    def task(obj, sym_vals_list: List[List[float]]) -> Tuple[List[NDArray]]:
        """Task converting list of symbol values into a meshgrid.

        :param sym_vals_list: List of values each symbol should take on
        the grid. There should be a list of values for each symbol.
        :return: Task converting a list of values in a meshgrid
        """

        return (
            [
                np.meshgrid(*sym_val_list, indexing="ij")  # type: ignore
                for sym_val_list in sym_vals_list
            ],
        )

    return MitTask(
        _label="ParamGridGen",
        _n_out_wires=1,
        _n_in_wires=1,
        _method=task,
    )


def gen_spectral_filtering_MitEx(backend: Backend, n_vals: int, **kwargs) -> MitEx:
    """Generator function for the spectral filtering MitEx. This method
    acts on symbolic circuits, evaluating the circuit on a grid of symbol
    values and performing mitigation on the resulting landscape.

    :param backend: Backend on which all experiments are run.
    :param n_vals: The number of values each symbol should take. Each symbol
        will take n_vals equally spaced values in the range
        :math:`[0, 2 U+03C0]`. The circuit will be evaluate at every
        permutation of the symbols evaluated at these points, giving a grid
        of circuit evaluations.

    :key signal_filter: Method for filtering the landscape of circuit
        evaluations. Defaults to SmallCoefficientSignalFilter.

    :return: MitEx implementing spectral filtering.
    """

    _optimisation_level = kwargs.get("optimisation_level", 0)

    _experiment_mitres = copy(
        kwargs.get(
            "experiment_mitres",
            gen_compiled_MitRes(backend, optimisation_level=_optimisation_level),
        )
    )

    _experiment_mitex = copy(
        kwargs.get(
            "experiment_mitex",
            MitEx(backend, _label="ExperimentMitex", mitres=_experiment_mitres),
        )
    )

    characterisation_taskgraph = TaskGraph().from_TaskGraph(_experiment_mitex)
    characterisation_taskgraph.add_n_wires(2)
    characterisation_taskgraph.prepend(gen_flatten_task())
    characterisation_taskgraph.append(gen_reshape_task())

    characterisation_taskgraph.prepend(gen_obs_exp_grid_gen_task())

    param_grid_gen_task = TaskGraph(_label="ParamGridGen")
    param_grid_gen_task.parallel(gen_param_grid_gen_task())
    characterisation_taskgraph.prepend(param_grid_gen_task)

    characterisation_taskgraph.append(gen_ndarray_to_dict_task())
    characterisation_taskgraph.append(gen_fft_task())

    signal_filter = kwargs.get("signal_filter", SmallCoefficientSignalFilter(tol=5))

    characterisation_taskgraph.append(gen_mitigation_task(signal_filter=signal_filter))

    characterisation_taskgraph.append(gen_inv_fft_task())

    characterisation_taskgraph.add_n_wires(2)
    characterisation_taskgraph.prepend(gen_wire_copy_task(2, 2))
    characterisation_taskgraph.prepend(gen_symbol_val_gen_task(n_sym_vals=n_vals))
    characterisation_taskgraph.append(gen_result_extraction_task())

    return MitEx(backend).from_TaskGraph(characterisation_taskgraph)

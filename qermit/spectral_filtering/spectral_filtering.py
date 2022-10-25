import numpy as np
from scipy import fft, interpolate  # type: ignore
from itertools import product
import matplotlib.pyplot as plt  # type: ignore
from typing import Tuple, List, Union
from qermit.taskgraph.task_graph import TaskGraph
from qermit.taskgraph.mittask import (
    MitTask,
    CircuitShots,
    Wire,
)
from qermit import SymbolsDict, AnsatzCircuit, ObservableExperiment
from qermit.taskgraph.mitex import MitEx, gen_compiled_MitRes
from pytket.backends import Backend
from pytket.utils import QubitPauliOperator
from copy import copy, deepcopy
from .results_cache import SpectralFilteringCache
from .signal_filter import SmallCoefficientSignalFilter
from numpy.typing import NDArray


# TODO: This should be replaced by an approach using the fourier
# coefficients directly.
def gen_result_extraction_task():
    
    def task(obj, result_list, obs_exp_list, points_list) -> Tuple[List[QubitPauliOperator]]:

        interp_result_list = []
        for result, points, obs_exp in zip(result_list, points_list, obs_exp_list):
            interp_point = list(obs_exp.AnsatzCircuit.SymbolsDict._symbolic_map.values())
            interp_qpo = deepcopy(obs_exp.ObservableTracker.qubit_pauli_operator)
            # TODO: I have assumed just one string in the observable here which is, of course, wrong.
            for qps in interp_qpo._dict.keys():
                interp_qpo._dict[qps] = interpolate.interpn(points, result[qps], interp_point)[0]
            interp_result_list.append(interp_qpo)

        return (interp_result_list, )
    
    return MitTask(_label="ResultExtraction", _n_out_wires=1, _n_in_wires=3, _method=task)

def gen_mitigation_task(signal_filter):
    
    def task(obj, fft_result_val_grid_list):

        mitigated_fft_result_val_grid_list = []
        for fft_result_val_grid in fft_result_val_grid_list:
            mitigated_fft_result_val_grid_dict = dict()
            for key, val in fft_result_val_grid.items():
                mitigated_fft_result_val_grid_dict[key] = signal_filter.filter(val)
            mitigated_fft_result_val_grid_list.append(mitigated_fft_result_val_grid_dict)
        
        return (mitigated_fft_result_val_grid_list, )
    
    return MitTask(_label="Mitigation", _n_out_wires=1, _n_in_wires=1, _method=task)

# TODO: the task which extracts the results from the qubit pauli operators
# should be separated so that the FFT step can be used more generally.
def gen_fft_task():
    
    def task(obj, result_grid_list):
        
        fft_result_grid_list = []
        
        for qpo_result_grid in result_grid_list:

            zero_qpo_result_grid = qpo_result_grid[tuple(0 for _ in qpo_result_grid.shape)]
            result_grid_dict = dict()
            for key in zero_qpo_result_grid._dict.keys():
                result_grid_dict[key] = np.empty(qpo_result_grid.shape, dtype=float)
                                    
            grid_point_val_list = [[i for i in range(size)] for size in qpo_result_grid.shape]
            for grid_point in product(*grid_point_val_list):
                qpo_result_dict = qpo_result_grid[grid_point]._dict
                for key, val in qpo_result_dict.items():
                    result_grid_dict[key][grid_point] = val
                                      
            fft_result_grid_dict = dict()
            for key, val in result_grid_dict.items():
                fft_result_grid_dict[key] = fft.fftn(val)
            fft_result_grid_list.append(fft_result_grid_dict)
            
        return (fft_result_grid_list, )
    
    return MitTask(_label="FFT", _n_out_wires=1, _n_in_wires=1, _method=task)

def gen_inv_fft_task():
    
    def task(obj, mitigated_fft_result_val_grid_list):
        
        # Iterate through results and invert FFT
        mitigated_result_val_grid_list = []
        for mitigated_fft_result_val_grid_dict in mitigated_fft_result_val_grid_list:
            mitigated_result_val_grid_dict = dict()
            for key, val in mitigated_fft_result_val_grid_dict.items():
                mitigated_result_val_grid_dict[key] = fft.ifftn(val)
            mitigated_result_val_grid_list.append(mitigated_result_val_grid_dict)
            
        return (mitigated_result_val_grid_list, )
    
    return MitTask(_label="InvFFT", _n_out_wires=1, _n_in_wires=1, _method=task)


def gen_flatten_task():
    
    def task(obj, grid_list:list[NDArray]) -> tuple[list[ObservableExperiment], list[int], list[tuple[int]]]:
        
        # Store structure of flattened grid as list of dictionaries.
        shape_list = [grid.shape for grid in grid_list]
        length_list = []
        # ObservableExperiments, currently stored in a grid, are flattened to a single list.
        flattened_grid_list = []
        for grid in grid_list:
            flattened_grid = grid.flatten()
            length_list.append(len(flattened_grid))
            flattened_grid_list += list(flattened_grid)
                                            
        return (flattened_grid_list, length_list, shape_list, )
    
    return MitTask(_label="Flatten", _n_out_wires=3, _n_in_wires=1, _method=task)

def gen_reshape_task():
    
    def task(obj, result_list, length_list, shape_list) -> Tuple[List[np.ndarray[QubitPauliOperator]]]:
        
        result_grid_list = []

        for length, shape in zip(length_list, shape_list):
            flattened_result_grid = result_list[:length]
            del result_list[:length]
            result_grid = np.reshape(flattened_result_grid, shape)
            result_grid_list.append(result_grid)
                
        return (result_grid_list, )
    
    return MitTask(_label="Reshape", _n_out_wires=1, _n_in_wires=3, _method=task)


def gen_obs_exp_grid_gen_task() -> MitTask:
    """Generates task creating an ObservableExperiment for each point in the 
    inputted meshgrid.

    :return: Task generating grid of experiments.
    :rtype: MitTask
    """
    
    def task(
        obj,
        obs_exp_list:list[ObservableExperiment],
        obs_exp_sym_val_grid_list:list[NDArray[float]]
    ) -> tuple[list[NDArray[ObservableExperiment]]]:
        """Task generating a grid of observable experiments.
        Each point in the grid corresponds to substituting the symbols in
        the circuit of each `ObservableExperiment` for a value in the
        meshgrids of `obs_exp_sym_val_grid_list`.

        :param obs_exp_list: A list of `ObservableExperiments`.
        :type obs_exp_list: list[ObservableExperiment]
        :param obs_exp_sym_val_grid_list: A list of collections of meshgrids.
        Each symbol in the Circuit of each `ObservableExperiment` is
        represented in the meshgrids of `obs_exp_sym_val_grid_list`.
        :type obs_exp_sym_val_grid_list: list[NDArray[float]]
        :return: A grid of `ObservableExperiments`. Each `ObservableExperiment`
        includes a `SymbolDict` evaluated at a point in the grid of
        `obs_exp_sym_val_grid_list`.
        :rtype: tuple[list[NDArray[ObservableExperiment]]]
        """
        
        # A grid of ObservableExperiment is generated for each ObservableExperiment in obs_exp_list
        obs_exp_grid_list = []
        for obs_exp, sym_val_grid_list in zip(obs_exp_list, obs_exp_sym_val_grid_list):
                      
            # List of symbols in circuit
            sym_list = list(obs_exp.AnsatzCircuit.SymbolsDict.symbols_list)
            # Initialise empty grid of ObservableExperiment
            obs_exp_grid = np.empty(sym_val_grid_list[0].shape, dtype=ObservableExperiment)
                                    
            # Generate an ObservableExperiment for every symbol value in the grid
            grid_point_val_list = [[i for i in range(size)] for size in sym_val_grid_list[0].shape]
            for grid_point in product(*grid_point_val_list):
                                                
                # Generate dictionary mapping every symbol to it's value at the given point in the grid.
                sym_map = {sym:sym_val_grid[grid_point] for sym_val_grid, sym in zip(sym_val_grid_list, sym_list)}  
                sym_dict = SymbolsDict().symbols_from_dict(sym_map)
                
                circ = obs_exp.AnsatzCircuit.Circuit.copy()

                anz_circ = AnsatzCircuit(circ, obs_exp.AnsatzCircuit.Shots, sym_dict)
                obs = deepcopy(obs_exp.ObservableTracker) # This needs to be a deep copy
                
                obs_exp_grid[grid_point] = ObservableExperiment(anz_circ, obs)
                                
            obs_exp_grid_list.append(obs_exp_grid)
        
        return (obs_exp_grid_list, )
    
    return MitTask(_label="ObsExpGridGen", _n_out_wires=1, _n_in_wires=2, _method=task)


def gen_symbol_val_gen_task(n_sym_vals:int) -> MitTask:
    """Generates task which produces a grid of values taken by the symbols in
    the circuit. The values are generated uniformly in the interval [0,2]
    (factors of pi give full coverage) for each symbol. The points on the
    grid are equally spaces, and the number of values is the same in
    each dimension.

    :param n_sym_vals: The number of values to be taken by each symbol in the circuit.
    :type n_sym_vals: int
    :return: Task generating grid of symbol values.
    :rtype: MitTask
    """
    
    def task(obj, obs_exp_list: List[ObservableExperiment]) -> Tuple[List[ObservableExperiment], List[List[np.ndarray]]]:
        """Produces a grid of values taken by the symbols in
        the circuit. The values are generated uniformly in the interval [0,2]
        (factors of pi give full coverage) for each symbol. The points on the
        grid are equally spaces, and the number of values is the same in
        each dimension.

        :param obs_exp_list: List of ObservableExperiments to be run.
        :type obs_exp_list: List[ObservableExperiment]
        :return: List of observable experiments is unchanged. For each
            experiment, a grid is generated.
        :rtype: Tuple[List[ObservableExperiment], List[List[np.ndarray]]]
        """
        
        # A collection of symbol value lists is generated for each
        # ObservableExperiment in obs_exp_list. Within each collection of
        # lists, there is a list for each of the symbols.
        sym_vals_list = []
        for obs_exp in obs_exp_list:
                        
            # List of symbols used in circuit
            sym_list = obs_exp.AnsatzCircuit.SymbolsDict.symbols_list
            
            # Lists of values taken by symbols, in half rotations
            sym_vals = [np.linspace(0, 2, n_sym_vals, endpoint=False) for _ in sym_list]
            sym_vals_list.append(sym_vals)
        
        return (obs_exp_list, sym_vals_list, )
    
    return MitTask(_label="SymbolValGen", _n_out_wires=2, _n_in_wires=1, _method=task)

# TODO: This task should be moved to somthing like a utilities folder,
# if it does not already exist.
def gen_wire_copy_task(n_in_wires:int, n_wire_copies:int) -> MitTask:
    """Generates task which copies each of the input wires `n_wire_copies`
    times. The output wires are repeated in the same order as the input wires.
    This is to say that input wires 1,2,...n will be outputted as
    1,2,...,n,...,1,2,...n.

    :param n_in_wires: The number of wires input into this task.
    :type n_in_wires: int
    :param n_wire_copies: The number of times the wires are to be copied.
    :type n_wire_copies: int
    :return: Task which coppies input wires.
    :rtype: MitTask
    """

    # TODO: I'm not sure what to do about the typing of this function.
    def task(obj, *in_wires):
        """Task copying `in_wires`. The output wires are repeated in the
        same order as the input wires. This is to say that input wires
        1,2,...n will be outputted as 1,2,...,n,...,1,2,...n.

        :return: Coppied wires
        """
        out_wires = tuple()
        for _ in range(n_wire_copies):
            for wire in in_wires:
                out_wires = (*out_wires, copy(wire))
        return out_wires

    return MitTask(_label="WireCopy", _n_out_wires=n_in_wires*n_wire_copies, _n_in_wires=n_in_wires, _method=task)

def gen_param_grid_gen_task() -> MitTask:
    """Generates task converting lists of symbol values into a mesgrid.

    :return: Task converting list of symbol values into a meshgrid.
    :rtype: MitTask
    """

    def task(obj, sym_vals_list:list[list[float]]) -> Tuple[list[list]]:
        """Task converting list of symbol values into a meshgrid.

        :param sym_vals_list: List of values each symbol should take on
        the grid. There should be a list of values for each symbol.
        :type sym_vals_list: list[list[float]]
        :return: Task converting a list of values in a meshgrid
        :rtype: Tuple[list[list]]
        """

        obs_exp_sym_val_grid_list = []
        for sym_val_list in sym_vals_list:

            # Grid corresponding to symbol values
            sym_val_grid_list = np.meshgrid(*sym_val_list, indexing='ij')
            obs_exp_sym_val_grid_list.append(sym_val_grid_list)

        return (obs_exp_sym_val_grid_list, )

    return MitTask(_label="ParamGridGen", _n_out_wires=1, _n_in_wires=1, _method=task)


def gen_spectral_filtering_MitEx(backend:Backend, n_vals:int, **kwargs) -> MitEx:

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
    
    _experiment_taskgraph = TaskGraph().from_TaskGraph(_experiment_mitex)

    # cache = kwargs.get("cache", None)
    # if cache:
    #     cache.n_vals = n_vals
    
    characterisation_taskgraph = TaskGraph().from_TaskGraph(_experiment_mitex)
    characterisation_taskgraph.add_wire()
    characterisation_taskgraph.add_wire()
    characterisation_taskgraph.prepend(gen_flatten_task())
    characterisation_taskgraph.append(gen_reshape_task())

    characterisation_taskgraph.prepend(gen_obs_exp_grid_gen_task())
    
    # TODO: Is this the neatest way to do this? I'm basically trying to add a
    # wire to the left of `param_grid_gen_task`.
    param_grid_gen_task = TaskGraph()
    param_grid_gen_task.parallel(gen_param_grid_gen_task())
    characterisation_taskgraph.prepend(param_grid_gen_task)

    characterisation_taskgraph.append(gen_fft_task())

    signal_filter = kwargs.get("signal_filter", SmallCoefficientSignalFilter(tol=5))

    characterisation_taskgraph.append(gen_mitigation_task(signal_filter=signal_filter))

    characterisation_taskgraph.append(gen_inv_fft_task())

    experiment_taskgraph = characterisation_taskgraph
    
    experiment_taskgraph.add_wire()
    experiment_taskgraph.add_wire()
    experiment_taskgraph.prepend(gen_wire_copy_task(2, 2))
    experiment_taskgraph.prepend(gen_symbol_val_gen_task(n_sym_vals=n_vals))
    experiment_taskgraph.append(gen_result_extraction_task())

    return MitEx(_experiment_mitex).from_TaskGraph(experiment_taskgraph)
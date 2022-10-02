import numpy as np
from scipy import fft
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
from abc import ABC
from scipy.interpolate import interpn

def my_plot(z, *axis):
    
    if len(axis)==2:
        plot_3d(*axis, z)
    elif len(axis)==1:
        plot_2d(*axis, z)

def plot_3d(x,y,z):
    
    fig = plt.figure()
    
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.scatter(x,y, np.real(z))
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.scatter(x,y, np.imag(z))
    
    plt.show()
    
def plot_2d(x,y):
    
    fig = plt.figure()
    
    ax = fig.add_subplot(1, 2, 1)
    ax.scatter(x, np.real(y))
    ax = fig.add_subplot(1, 2, 2)
    ax.scatter(x, np.imag(y))
    
    plt.show()

class SpectralFilteringCache:

    def __init__(self):
        self.n_vals = None 
        self.obs_exp_sym_val_grid_list = None
        self.mitigated_fft_result_val_grid_list = None
        self.ifft_mitigated_result_val_grid_list = None
        self.fft_result_grid_list = None
        self.result_grid_list = None

    def plot_mitigated_fft(self):

        for fft_result_val_grid, param_grid in zip(self.mitigated_fft_result_val_grid_list, self.obs_exp_sym_val_grid_list):

            xf = fft.fftfreq(self.n_vals, 1 / self.n_vals)                                    
            axis = np.meshgrid(*[xf for _ in range(len(param_grid))])
            my_plot(fft_result_val_grid, *axis)

    def plot_mitigated_ifft(self):

        for mitigated_result_val_grid, param_grid in zip(self.ifft_mitigated_result_val_grid_list, self.obs_exp_sym_val_grid_list):                    
            my_plot(mitigated_result_val_grid, *param_grid)

    def plot_fft_result_grid(self):

        for fft_result_grid, param_grid in zip(self.fft_result_grid_list, self.obs_exp_sym_val_grid_list):
            xf = fft.fftfreq(self.n_vals, 1 / self.n_vals)                                    
            axis = np.meshgrid(*[xf for _ in range(len(param_grid))])
            my_plot(fft_result_grid, *axis)

    def plot_result_grid(self):

        for result_grid, param_grid in zip(self.result_grid_list, self.obs_exp_sym_val_grid_list):                                    
            my_plot(result_grid, *param_grid)


def gen_result_extraction_task():
    
    def task(obj, result_list, points_list, obs_exp_list) -> Tuple[List[QubitPauliOperator]]:

        interp_result_list = []
        for result, points, obs_exp in zip(result_list, points_list, obs_exp_list):
            interp_point = list(obs_exp.AnsatzCircuit.SymbolsDict._symbolic_map.values())
            interp_qpo = deepcopy(obs_exp.ObservableTracker.qubit_pauli_operator)
            # TODO: I have assumed just on string in the observable here which is, of course, wrong.
            for qps in interp_qpo._dict.keys():
                interp_qpo._dict[qps] = interpn(points, result, interp_point)[0]
            interp_result_list.append(interp_qpo)

        return (interp_result_list, )
    
    return MitTask(_label="ResultExtraction", _n_out_wires=1, _n_in_wires=3, _method=task)

class SignalFilter(ABC):

    def filter(self, fft_result_val_grid):
        pass

class SmallCoefficientSignalFilter(SignalFilter):

    def __init__(self, tol):
        self.tol = tol

    def filter(self, fft_result_val_grid):

        mitigated_fft_result_val_grid = deepcopy(fft_result_val_grid)
        mitigated_fft_result_val_grid[np.abs(mitigated_fft_result_val_grid) < self.tol] = 0.0
        return mitigated_fft_result_val_grid

def gen_mitigation_task(cache, signal_filter):
    
    def task(obj, fft_result_val_grid_list):

        mitigated_fft_result_val_grid_list = []
        for fft_result_val_grid in fft_result_val_grid_list:
            mitigated_fft_result_val_grid_list.append(signal_filter.filter(fft_result_val_grid))
                    
        if cache:
            cache.mitigated_fft_result_val_grid_list = deepcopy(mitigated_fft_result_val_grid_list)
        
        return (mitigated_fft_result_val_grid_list, )
    
    return MitTask(_label="Mitigation", _n_out_wires=1, _n_in_wires=1, _method=task)

def gen_fft_task(cache=None):
    
    def task(obj, result_grid_list):
        
        fft_result_grid_list = []
        float_result_grid_list = []
        
        for qpo_result_grid in result_grid_list:
                                    
            result_grid = np.empty(qpo_result_grid.shape, dtype=float) 
            grid_point_val_list = [[i for i in range(size)] for size in qpo_result_grid.shape]
            for grid_point in product(*grid_point_val_list):
                qpo_result_dict = qpo_result_grid[grid_point]._dict
                result = qpo_result_dict[list(qpo_result_dict.keys())[0]]
                result_grid[grid_point] = result
                                      
            # Perform FFT on grid of results.
            fft_result_grid = fft.fftn(result_grid) 
            fft_result_grid_list.append(fft_result_grid)

            float_result_grid_list.append(result_grid)
                        
        if cache:
            cache.fft_result_grid_list = deepcopy(fft_result_grid_list)
            cache.result_grid_list = deepcopy(float_result_grid_list)
            
        return (fft_result_grid_list, )
    
    return MitTask(_label="FFT", _n_out_wires=1, _n_in_wires=1, _method=task)

def gen_inv_fft_task(cache):
    
    def task(obj, mitigated_fft_result_val_grid_list):
        
        # Iterate through results and invert FFT
        mitigated_result_val_grid_list = []
        for mitigated_fft_result_val_grid in mitigated_fft_result_val_grid_list:
            mitigated_result_val_grid = fft.ifftn(mitigated_fft_result_val_grid)
            mitigated_result_val_grid_list.append(mitigated_result_val_grid)
                
        if cache:
            cache.ifft_mitigated_result_val_grid_list = deepcopy(mitigated_result_val_grid_list)
            
        return (mitigated_result_val_grid_list, )
    
    return MitTask(_label="InvFFT", _n_out_wires=1, _n_in_wires=1, _method=task)


def gen_flatten_task():
    
    def task(obj, obs_exp_grid_list):
        
        # ObservableExperiments, currently stored in a grid, are flattened to a single list.
        flattened_obs_exp_grid_list = []
        # Store structure of flattened grid as list of dictionaries.
        length_list = []
        shape_list = []
                
        for obs_exp_grid in obs_exp_grid_list:
            
            obs_exp_grid_shape = obs_exp_grid.shape
            shape_list.append(obs_exp_grid_shape)
            flattened_obs_exp_grid = obs_exp_grid.flatten()
            flattened_obs_exp_grid_len = len(flattened_obs_exp_grid)
            length_list.append(flattened_obs_exp_grid_len)
            flattened_obs_exp_grid_list += list(flattened_obs_exp_grid)
                                            
        return (flattened_obs_exp_grid_list, length_list, shape_list, )
    
    return MitTask(_label="Flatten", _n_out_wires=3, _n_in_wires=1, _method=task)

def gen_reshape_task(cache):
    
    def task(obj, result_list, length_list, shape_list) -> Tuple[List[QubitPauliOperator]]:
        
        result_grid_list = []

        for length, shape in zip(length_list, shape_list):
            flattened_result_grid = result_list[:length]
            del result_list[:length]
            result_grid = np.reshape(flattened_result_grid, shape)
            result_grid_list.append(result_grid)
                
        return (result_grid_list, )
    
    return MitTask(_label="Reshape", _n_out_wires=1, _n_in_wires=3, _method=task)


def gen_obs_exp_grid_gen_task() -> MitTask:
    
    def task(obj, obs_exp_list, obs_exp_sym_val_grid_list):
        
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
                obs = deepcopy(obs_exp.ObservableTracker) # This needs to be a deep copy, which is a little scary
                
                obs_exp_grid[grid_point] = ObservableExperiment(anz_circ, obs)
                                
            obs_exp_grid_list.append(obs_exp_grid)
        
        return (obs_exp_grid_list, )
    
    return MitTask(_label="ObsExpGridGen", _n_out_wires=1, _n_in_wires=2, _method=task)


def gen_param_grid_gen_task(n_sym_vals:int, cache:Union[None, SpectralFilteringCache]) -> MitTask:
    """Generates task which produces a grid of values taken by the symbols in
    the circuit. The values are generated uniformly in the interval [0,2]
    (factors of pi give full coverage) for each symbol.

    :param n_sym_vals: The number of values to be taken by each symbol in the circuit
    :type n_sym_vals: int
    :return: Task generating grid of symbol values.
    :rtype: MitTask
    """
    
    def task(obj, obs_exp_list: List[ObservableExperiment]) -> Tuple[List[ObservableExperiment], List[List[np.ndarray]]]:
        
        # A symbol value grid is generated for each ObservableExperiment in obs_exp_list
        obs_exp_sym_val_grid_list = []
        sym_val_list_list = []
        for obs_exp in obs_exp_list:
                        
            # List of symbols used in circuit
            sym_list = obs_exp.AnsatzCircuit.SymbolsDict.symbols_list
            
            # Lists of values taken by symbols, in half rotations
            sym_val_list = [np.linspace(0, 2, n_sym_vals, endpoint=False) for _ in sym_list]
            sym_val_list_list.append(sym_val_list)
            
            # Grid corresponding to symbol values
            sym_val_grid_list = np.meshgrid(*sym_val_list)
            obs_exp_sym_val_grid_list.append(sym_val_grid_list)

        if cache:
            cache.obs_exp_sym_val_grid_list = obs_exp_sym_val_grid_list
                    
        return (obs_exp_list, obs_exp_sym_val_grid_list, sym_val_list_list, obs_exp_list, )
    
    return MitTask(_label="ParamGridGen", _n_out_wires=4, _n_in_wires=1, _method=task)


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

    cache = kwargs.get("cache", None)
    if cache:
        cache.n_vals = n_vals

    experiment_taskgraph = TaskGraph().from_TaskGraph(_experiment_mitex)
    experiment_taskgraph.add_wire()
    experiment_taskgraph.add_wire()
    experiment_taskgraph.prepend(gen_flatten_task())
    experiment_taskgraph.append(gen_reshape_task(cache=cache))

    experiment_taskgraph.prepend(gen_obs_exp_grid_gen_task())

    experiment_taskgraph.append(gen_fft_task(cache=cache))

    signal_filter = kwargs.get("signal_filter", SmallCoefficientSignalFilter(tol=5))

    experiment_taskgraph.append(gen_mitigation_task(cache=cache, signal_filter=signal_filter))

    experiment_taskgraph.append(gen_inv_fft_task(cache=cache))
    
    experiment_taskgraph.add_wire()
    experiment_taskgraph.add_wire()
    experiment_taskgraph.prepend(gen_param_grid_gen_task(n_sym_vals=n_vals, cache=cache))
    experiment_taskgraph.append(gen_result_extraction_task())

    return MitEx(_experiment_mitex).from_TaskGraph(experiment_taskgraph)
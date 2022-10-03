from abc import ABC
import numpy as np
from copy import copy, deepcopy

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
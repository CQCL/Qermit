from abc import ABC
import numpy as np
from copy import deepcopy
from numpy.typing import NDArray


class SignalFilter(ABC):
    """Base class for signal filtering."""

    def filter(self, fft_result_val_grid: NDArray[np.float64]) -> NDArray[np.float64]:
        """Method transforming array of floats into filtered array of
        floats."""
        pass


class SmallCoefficientSignalFilter(SignalFilter):
    """Child class of SignalFilter which filters results by reducing small
    Fourier coefficients to 0.
    """

    def __init__(self, tol: float):
        """Initialisation method.

        :param tol: Value below which coefficients should be reduced to 0
        :type tol: float
        """
        self.tol = tol

    def filter(self, fft_result_val_grid: NDArray[np.float64]) -> NDArray[np.float64]:
        """Filter method reducing values in `fft_result_val_grid` to 0 if
        they are less than `tol`.

        :param fft_result_val_grid: Grid of values.
        :type fft_result_val_grid: NDArray[float]
        :return: Grid of values with values less than `tol` reduced to 0.
        :rtype: NDArray[float]
        """

        mitigated_fft_result_val_grid = deepcopy(fft_result_val_grid)
        mitigated_fft_result_val_grid[
            np.abs(mitigated_fft_result_val_grid) < self.tol
        ] = 0.0
        return mitigated_fft_result_val_grid

import numpy as np
import matplotlib.pyplot as plt
from scipy import fft

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
import numpy as np
from qermit.taskgraph.mittask import MitTask

class SpectralFilteringCharacCache:

    def __init__(self):
        self.characterised = False

    def save_inputs(self, obs_exp_list, sym_val_list_list):

        self.obs_exp_list = obs_exp_list
        self.sym_val_list_list = sym_val_list_list

    def save_output(self, mitigated_result_val_grid_list):

        self.mitigated_result_val_grid_list = mitigated_result_val_grid_list
        self.characterised = True


def gen_initialise_characterisation_cache_task(charac_cache):

    def task(obj, obs_exp_list, sym_val_list_list):
        charac_cache.save_inputs(obs_exp_list, sym_val_list_list)
        return(obs_exp_list, sym_val_list_list, )

    return MitTask(_label="InitCharacCache", _n_out_wires=2, _n_in_wires=2, _method=task)

def gen_save_characterisation_cache_task(charac_cache):

    def task(obj, mitigated_result_val_grid_list):
        charac_cache.save_output(mitigated_result_val_grid_list)
        return(mitigated_result_val_grid_list, )

    return MitTask(_label="SaveCharacCache", _n_out_wires=1, _n_in_wires=1, _method=task)

def gen_regurgitate_cache_task(charac_cache):

    def task(obj, obs_exp_list, sym_val_list_list):

        assert charac_cache.obs_exp_list == obs_exp_list
        assert np.array_equal(charac_cache.sym_val_list_list, sym_val_list_list)

        return (charac_cache.mitigated_result_val_grid_list, )

    return MitTask(_label="RegurgitateCache", _n_out_wires=1, _n_in_wires=2, _method=task)
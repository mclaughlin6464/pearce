from pearce.emulator import OriginalRecipe, ExtraCrispy, SpicyBuffalo
from pearce.mocks import cat_dict
import numpy as np
from os import path
from SloppyJoes import lazy_wrapper

training_file = '/scratch/users/swmclau2/xi_zheng07_cosmo_lowmsat/PearceRedMagicXiCosmoFixedNd.hdf5'
em_method = 'gp'
fixed_params = {'z':0.0, 'r':24.06822623}
emu = OriginalRecipe(training_file, method = em_method, fixed_params=fixed_params,
                 custom_mean_function = 'linear', downsample_factor = 0.1)

def resids(p, gp, x, y):
    gp.set_parameter_vector(p)
    gp.recompute()
    return gp.predict(emu.downsample_y, x, return_cov=False)-y

vals = np.ones_like(emu._emulator.get_parameter_vector())
args = (emu._emulator, emu.x, emu.y)

result = lazy_wrapper(resids, vals, func_args = args, print_level = 3)\

print result
np.savetxt('sloppy_joes_result.npy', result)

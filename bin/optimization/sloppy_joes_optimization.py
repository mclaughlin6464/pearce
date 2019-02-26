from pearce.emulator import OriginalRecipe, ExtraCrispy, SpicyBuffalo, LemonPepperWet
from pearce.mocks import cat_dict
import numpy as np
from os import path
from SloppyJoes import lazy_wrapper

training_file = '/scratch/users/swmclau2/xi_zheng07_cosmo_lowmsat/PearceRedMagicXiCosmoFixedNd.hdf5'
em_method = 'gp'
fixed_params = {'z':0.0}#, 'r':24.06822623}
emu = LemonPepperWet(training_file, method = em_method, fixed_params=fixed_params,
                 custom_mean_function = 'linear', downsample_factor = 0.01)

def resids(p, gps, xs, ys, yerrs):
    res = []
    for gp, x, y,yerr, dy in zip(gps, xs, ys,yerrs, emu.downsample_y):
        gp.set_parameter_vector(p)
        gp.recompute()
        r = (gp.predict(dy, x, return_cov=False)-y)/(yerr+1e-5)
        res.append(r)

    #print res[0].shape
    return np.hstack(res) 

vals = np.ones_like(emu._emulators[0].get_parameter_vector())
args = (emu._emulators, emu.x, emu.y, emu.yerr)

result = lazy_wrapper(resids, vals, func_args = args, print_level = 3)\

print result
np.savetxt('sloppy_joes_result_lpw.npy', result)

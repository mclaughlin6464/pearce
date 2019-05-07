from pearce.emulator import OriginalRecipe, ExtraCrispy, SpicyBuffalo, LemonPepperWet
from pearce.mocks import cat_dict
import numpy as np
from os import path
from SloppyJoes import lazy_wrapper

training_file = '/scratch/users/swmclau2/xi_zheng07_cosmo_lowmsat/PearceRedMagicXiCosmoFixedNd.hdf5'
em_method = 'gp'
fixed_params = {'z':0.0, 'r': 0.19118072}
#emu = SpicyBuffalo(training_file, method = em_method, fixed_params=fixed_params,
#                 custom_mean_function = 'linear', downsample_factor = 0.01)
emu = OriginalRecipe(training_file, method = em_method, fixed_params=fixed_params,
                 custom_mean_function = 'linear', downsample_factor = 0.01)


def resids_bins(p, gps, xs, ys, yerrs):
    res = []
    p_np = np.array(p).reshape((len(gps), -1))
    for gp, x, y,yerr, dy, p in zip(gps, xs, ys,yerrs, emu.downsample_y, p_np):
        gp.set_parameter_vector(p)
        gp.recompute()
        r = (gp.predict(dy, x, return_cov=False)-y)/(yerr+1e-5)
        res.append(r)

    #print res[0].shape
    return np.hstack(res) 

def resids(p, gp, x, y, yerr):
    p = np.array(p)
    gp.set_parameter_vector(p)
    gp.recompute()
    res = (gp.predict(emu.downsample_y, x, return_cov=False)-y)/(yerr+1e-5)

    #print res[0].shape
    return res

n_hps = len(emu._emulator.get_parameter_vector())

#vals = np.ones((n_hps*emu.n_bins))
vals = np.ones((n_hps,))
args = (emu._emulator, emu.x, emu.y, emu.yerr)

result = lazy_wrapper(resids, vals, func_args = args, print_level = 3)\

print result
np.savetxt('sloppy_joes_result_indiv_bins.npy', result)

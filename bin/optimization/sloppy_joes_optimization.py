from pearce.emulator import OriginalRecipe, ExtraCrispy
from pearce.mocks import cat_dict
import numpy as np
from os import path
training_dir = '/u/ki/swmclau2/des/PearceLHC_wp_z'
em_method = 'gp'
fixed_params = {'z':0.0}#, 'r':0.18477483}
emu = OriginalRecipe(training_dir, method = em_method, fixed_params=fixed_params)
emulation_point = [('f_c', 0.233), ('logM0', 12.0), ('sigma_logM', 0.533),
                    ('alpha', 1.083),('logM1', 13.5), ('logMmin', 12.233)]
em_params = dict(emulation_point)
em_params.update(fixed_params)
del em_params['z']

param_names = em_params.keys()

rp_bins =  list(np.logspace(-1,1.5,19) )
rp_bins.pop(1)
rp_bins = np.array(rp_bins)
rpoints =  (rp_bins[1:]+rp_bins[:-1])/2.0

from SloppyJoes import lazy_wrapper
def resids(p, gp, y):
    gp.kernel[:] = p
    gp.recompute()
    return gp.predict(y, gp._x, mean_only=True)-y

vals = np.ones_like(emu._emulator.kernel.vector)
args = (emu._emulator, emu.y)
result = lazy_wrapper(resids, vals, func_args = args, print_level = 3)\
             #artol = 1e-6, xrtol = 1e-6, xtol=1e-6, gtol = 1e-6)
print result
np.savetxt('/u/ki/swmclau2/Git/pearce/bin/result.npy', result)

from pearce.emulator import OriginalRecipe, ExtraCrispy
training_file = '/u/ki/swmclau2/des/PearceRedMagicWpCosmo.hdf5'

em_method = 'gp'
split_method = 'random'

a = 1.0
z = 1.0/a - 1.0

fixed_params = {'z':z, 'cosmo':0}#, 'r':0.18477483}

n_leaves, n_overlap = 20, 1
emu = ExtraCrispy(training_file, n_leaves, n_overlap, split_method, method = em_method, fixed_params=fixed_params)
print emu.get_param_names()

liklihoods, points = emu.hyperparam_random_grid_search(1000)

import numpy as np
np.savetxt('hod_lilklihoods.npy', liklihoods)
np.savetxt('hod_points.npy', points)

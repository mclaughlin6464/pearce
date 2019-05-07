from pearce.emulator import OriginalRecipe, ExtraCrispy
training_file = '/u/ki/swmclau2/des/PearceRedMagicWpCosmo.hdf5'

em_method = 'gp'
split_method = 'random'

a = 1.0
z = 1.0/a - 1.0

fixed_params = {'z':z}#, 'r':0.18477483}

n_leaves, n_overlap = 500, 1
emu = ExtraCrispy(training_file, n_leaves, n_overlap, split_method, method = em_method, fixed_params=fixed_params)


liklihoods, points = emu.hyperparam_random_grid_search(200)

import numpy as np
np.savetxt('lilklihoods.npy', liklihoods)
np.savetxt('points.npy', points)

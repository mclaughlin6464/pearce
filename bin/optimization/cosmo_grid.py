from pearce.emulator import OriginalRecipe, ExtraCrispy
training_file = '/u/ki/swmclau2/des/PearceRedMagicWpCosmo.hdf5'
import numpy as np


em_method = 'gp'
split_method = 'random'

a = 1.0
z = 1.0/a - 1.0

fixed_params = {'z':z,'r':24.06822623}

#n_leaves, n_overlap = 500, 1
emu = OriginalRecipe(training_file, method = em_method, fixed_params=fixed_params,\
                    custom_mean_function =None, downsample_factor = 0.2)


liklihoods, points = emu.hyperparam_random_grid_search(500)

np.savetxt('cosmo_liklihoods.npy', liklihoods)
np.savetxt('cosmo_points.npy', points)

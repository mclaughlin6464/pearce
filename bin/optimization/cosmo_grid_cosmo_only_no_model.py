from pearce.emulator import OriginalRecipe, ExtraCrispy
import numpy as np

training_file = '/home/users/swmclau2/scratch/PearceRedMagicWpCosmo.hdf5'

em_method = 'gp'
split_method = 'random'

a = 1.0
z = 1.0/a - 1.0

fixed_params = {'z':z, 'HOD':0}

#n_leaves, n_overlap = 10, 2
emu = OriginalRecipe(training_file, method = em_method, fixed_params=fixed_params,\
                    custom_mean_function = None)


liklihoods, points = emu.hyperparam_random_grid_search(50000)

np.savetxt('cosmo_liklihoods_cosmo_no.npy', liklihoods)
np.savetxt('cosmo_points_cosmo_no.npy', points)

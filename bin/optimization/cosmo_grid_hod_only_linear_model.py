from pearce.emulator import OriginalRecipe, ExtraCrispy
import numpy as np

training_file = '/home/users/swmclau2/scratch/PearceRedMagicWpCosmo.hdf5'

em_method = 'gp'
split_method = 'random'

a = 1.0
z = 1.0/a - 1.0

fixed_params = {'z':z, 'cosmo':0}

n_leaves, n_overlap = 10, 2
emu = ExtraCrispy(training_file, n_leaves, n_overlap, split_method, method = em_method, fixed_params=fixed_params,\
                    custom_mean_function = 'linear')


liklihoods, points = emu.hyperparam_random_grid_search(5000)

np.savetxt('cosmo_liklihoods_hod_linear.npy', liklihoods)
np.savetxt('cosmo_points_hod_linear.npy', points)

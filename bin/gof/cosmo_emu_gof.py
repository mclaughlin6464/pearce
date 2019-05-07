from pearce.emulator import ExtraCrispy
import numpy as np

training_file = '/u/ki/swmclau2/des/PearceRedMagicWpCosmo.hdf5'

em_method = 'gp'
split_method = 'random'

a = 1.0
z = 1.0/a - 1.0

fixed_params = {'z':z}

n_leaves, n_overlap = 400, 2
emu = ExtraCrispy(training_file, n_leaves, n_overlap, split_method, method = em_method, fixed_params=fixed_params,
                         custom_mean_function = None)

truth_file = '/u/ki/swmclau2/des/PearceRedMagicWpCosmoTest.hdf5'
N = 100
gof = emu.goodness_of_fit(truth_file, N,  statistic = 'log_frac')

print n_leaves, n_overlap, N
print gof.mean(axis = 0) 
np.savetxt('gof_%d_%d.npy'%(n_leaves, n_overlap), gof)

from pearce.emulator import OriginalRecipe, ExtraCrispy
import numpy as np

training_file = '/home/users/swmclau2/scratch/PearceRedMagicWpCosmo.hdf5'

em_method = 'gp'
split_method = 'random'

a = 1.0
z = 1.0/a - 1.0

fixed_params = {'z':z, 'cosmo': 1}#, 'r':0.18477483}

n_leaves, n_overlap = 5, 2
emu = ExtraCrispy(training_file,n_leaves, n_overlap, split_method,  method = em_method, fixed_params=fixed_params,\
                    custom_mean_function = None)


results = emu.train_metric()

print results
print
print dict(zip(emu.get_param_names(), np.exp(results.x)))

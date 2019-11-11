from pearce.emulator import OriginalRecipe, ExtraCrispy
import numpy as np

training_file = '/home/users/swmclau2/scratch/PearceRedMagicWpCosmo.hdf5'

em_method = 'gp'
split_method = 'random'

a = 1.0
z = 1.0/a - 1.0

fixed_params = {'z':z, 'HOD': 19}#, 'r':0.18477483}

#n_leaves, n_overlap = 200, 1
emu = OriginalRecipe(training_file, method = em_method, fixed_params=fixed_params,\
                    custom_mean_function = None)


results = emu.train_metric()

print results
print
print dict(zip(emu.get_param_names(), np.exp(results.x)))

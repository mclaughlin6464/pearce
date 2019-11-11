from pearce.emulator import OriginalRecipe, ExtraCrispy

training_file = '/home/users/swmclau2/scratch/PearceRedMagicWpCosmo2.hdf5'

em_method = 'gp'
split_method = 'random'

a = 1.0
z = 1.0/a - 1.0

fixed_params = {'z':z, 'r':0.19118072}#, 'r':0.18477483}

n_leaves, n_overlap = 200, 1
emu = OriginalRecipe(training_file, method = em_method, fixed_params=fixed_params,\
                    custom_mean_function = None, downsample_factor = 0.02)


print 'hi'

import numpy as np
results = emu.train_metric()
print fixed_params
print emu.get_param_names()
print np.exp(results.x)
print results


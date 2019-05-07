from pearce.emulator import OriginalRecipe, ExtraCrispy
import numpy as np

training_file = '/u/ki/swmclau2/des/xi_cosmo_trainer/PearceRedMagicXiCosmoFixedNd.hdf5'

em_method = 'gp'
split_method = 'random'

a = 1.0
z = 1.0/a - 1.0

fixed_params = {'z':z, 'r':24.06822623}

#n_leaves, n_overlap = 200, 1
emu = OriginalRecipe(training_file, method = em_method, fixed_params=fixed_params,\
                            custom_mean_function = None, downsample_factor = 0.05)

v = np.array([-12.0550382,   0.1054246,   0.2661017,   5.6407612,   0.2408568,
             1.1295944,   0.3643993,  11.5649985,   4.9071932,   4.7031938,
                     11.7621938,  10.6279446,   0.       ,  10.6161248,   1.8339794,
                              7.342365 ,  10.6371797,   2.2441632,  13.8155106,  11.3512804,
                                       3.1795786,   4.6846614,   5.0188608,   3.7658774,  -1.5383083])

emu._emulator.set_parameter_vector(v)
results = emu.train_metric()

print emu.get_param_names()
print np.exp(results.x)
print results

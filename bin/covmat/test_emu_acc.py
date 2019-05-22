from pearce.emulator import NashvilleHot
from GPy.kern import *
import numpy as np
from os import path

#training_file = '/home/users/swmclau2/scratch/xi_gg_zheng07_cosmo_v3/PearceXiggCosmo.hdf5'
#test_file = '/home/users/swmclau2/scratch/xi_gg_zheng07_cosmo_test_v3/PearceXiggCosmoTest.hdf5'

training_file = '/home/users/swmclau2/scratch/xi_gg_fastpm/PearceXiggFastPM.hdf5'

em_method = 'gp'
fixed_params = {'z': 0.5503876}
hyperparams = {'kernel': (Matern32(input_dim=8, ARD=True) + RBF(input_dim=8, ARD=True)+Bias(input_dim=8),
                           Matern32(input_dim=4, ARD=True) + RBF(input_dim=4, ARD=True)+Bias(input_dim=4) ), \
               'optimize': True}

emu = NashvilleHot(training_file, hyperparams=hyperparams,fixed_params = fixed_params, downsample_factor = 1.0)
emu._save_as_default_kernel()
from sys import exit
exit(0)
for idx, (k1, k2) in enumerate(emu._kernels):
    print idx, "*"*20
    print k1
    print k2

x1, x2, y, yerr, _, _ = emu.get_data(test_file, fixed_params = fixed_params)

py = np.zeros_like(y)

for cidx, cp in enumerate(x1):
    for hidx, hp in enumerate(x2):
        params = dict(zip(emu.get_param_names(), np.hstack([cp,hp])))
        py[:,cidx, hidx] = emu.emulate_wrt_r(params)[0]


acc1 = np.abs(10**py - 10**y)/10**y

print acc1.mean(axis=1)
print acc1.std(axis=1)
print '-'*20

pred_y, data_y = emu.goodness_of_fit(test_file, statistic = None)
#R = (10**pred_y - 10**data_y)
#R = (pred_y - data_y)
#cov = R.dot(R.T)/(R.shape[1]-1)

#np.save('xi_gg_nh_emu_cov.npy', cov)

acc = (np.abs(10**pred_y - 10**data_y)/(10**data_y))
print acc.mean(axis =1)
print acc.std(axis=1)

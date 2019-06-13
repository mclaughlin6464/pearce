from pearce.emulator import NashvilleHot
from GPy.kern import *
import numpy as np
from os import path

#training_file = '/home/users/swmclau2/scratch/xi_gg_zheng07_cosmo_v4/PearceXiggCosmo.hdf5'
#test_file = '/home/users/swmclau2/scratch/xi_gg_zheng07_cosmo_test_v4/PearceXiggCosmoTest.hdf5'
training_file = '/home/users/swmclau2/scratch/xi_gg_corrabzheng07_v2/PearceXiggCosmoCorrAB.hdf5'
test_file = '/home/users/swmclau2/scratch/xi_gg_corrabzheng07_test_v2/PearceXiggCosmoCorrABTest.hdf5'


em_method = 'gp'
fixed_params = {'z':0.0}
hyperparams = {} 

emu = NashvilleHot(training_file, hyperparams=hyperparams,fixed_params = fixed_params, downsample_factor = 1.0)

pred_y, data_y = emu.goodness_of_fit(test_file, statistic = None)
R = (10**pred_y - 10**data_y)
##R = (pred_y - data_y)
cov = R.dot(R.T)/(R.shape[1]-1)

np.save('xi_gg_nh_emu_corrab_cov_v4.npy', cov)

print (np.abs(10**pred_y - 10**data_y)/(10**data_y)).mean(axis =1)

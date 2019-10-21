from pearce.emulator import NashvilleHot
from GPy.kern import *
import numpy as np
from os import path
from sys import argv
import h5py

training_file = argv[1]
assert path.isfile(training_file) 
#test_file = argv[2]
#assert path.isfile(test_file) 

f = h5py.File(training_file, 'r')
HOD_params = len(f.attrs['hod_param_names'])
f.close()

fixed_params = {'z':0.0}
hyperparams = {'kernel': (Matern32(input_dim=7, ARD=True) + RBF(input_dim=7, ARD=True)+Bias(input_dim=7),
                           Matern32(input_dim=HOD_params, ARD=True) + RBF(input_dim=HOD_params, ARD=True)+Bias(input_dim=HOD_params) ), \
               'optimize': True}

#for df in [0.5]:#,0.25,  0.5]: 
emu = NashvilleHot(training_file, hyperparams=hyperparams,fixed_params = fixed_params)#, downsample_factor = df)
emu.save_as_default_kernel()

#pred_y, data_y = emu.goodness_of_fit(test_file, downsample_factor = df, statistic = None)

#print df, (np.abs(10**pred_y - 10**data_y)/(10**data_y)).mean(axis =1)


from pearce.emulator import NashvilleHot
from GPy.kern import *
import numpy as np
from os import path
from sys import argv
import h5py

training_file = argv[1]
assert path.isfile(training_file) 

f = h5py.File(training_file, 'r')
HOD_params = len(f.attrs['hod_param_names'])
f.close()

fixed_params = {'z':0.0}
hyperparams = {'kernel': (RBF(input_dim=7, ARD=True)+Bias(input_dim=7),
                           Matern32(input_dim=HOD_params, ARD=True) + RBF(input_dim=HOD_params, ARD=True)+Bias(input_dim=HOD_params) ), \
               'optimize': True}

emu = NashvilleHot(training_file, hyperparams=hyperparams,fixed_params = fixed_params, downsample_factor = 0.5)
emu.save_as_default_kernel()

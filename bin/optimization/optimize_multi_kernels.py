from pearce.emulator import NashvilleHot
from itertools import product
from GPy.kern import *
import numpy as np
from os import path
from sys import argv
import h5py

training_file = argv[1]
assert path.isfile(training_file) 
test_file = argv[2]
assert path.isfile(test_file)
save_fname = argv[3]

cosmo_idx, hod_idx = int(argv[4]),int(argv[5])

f = h5py.File(training_file, 'r')
HOD_params = len(f.attrs['hod_param_names'])
f.close()

fixed_params = {'z':0.0}

cosmo_kernels = [Linear(input_dim=7, ARD=True), RBF(input_dim=7, ARD=True), Linear(input_dim=7, ARD=True) + RBF(input_dim=7, ARD=True), Linear(input_dim=7, ARD=True) + Matern32(input_dim=7, ARD=True), \
          Matern32(input_dim=7, ARD=True)+RBF(input_dim=7, ARD=True) + Bias(input_dim=7)]
HOD_kernels = [ Matern32(input_dim=HOD_params, ARD=True), RBF(input_dim=HOD_params, ARD=True) + Linear(input_dim=HOD_params, ARD=True), Matern32(input_dim=HOD_params, ARD=True)+RBF(input_dim=HOD_params, ARD=True) + Bias(input_dim=HOD_params)]#, RBF(input_dim=HOD_params, ARD=True) + Matern32(input_dim=HOD_params, ARD=True)]

#k = (cosmo_kernels[3], HOD_kernels[0])
k = (cosmo_kernels[cosmo_idx], HOD_kernels[hod_idx])

# best for wp:  either   RBF & Matern32 or Linear + RBF & RBF & Matern32
# best for DS: Linear + RBF, Matern32
#kernels = product(cosmo_kernels, HOD_kernels) 
#for k in kernels:
hyperparams = {'kernel': k , \
               'optimize': True}

#for df in [0.5]:#,0.25,  0.5]: 
emu = NashvilleHot(training_file, hyperparams=hyperparams,fixed_params = fixed_params, downsample_factor = 0.5)
emu.save_as_default_kernel()
#emu = NashvilleHot(training_file, fixed_params = fixed_params)#, downsample_factor = df)
#
pred_y, data_y = emu.goodness_of_fit(test_file, statistic = None, downsample_factor = 0.1)

print 'Bias', ((10**pred_y - 10**data_y)/(10**data_y)).mean(axis=1)
print 'Acc',  (np.abs(10**pred_y - 10**data_y)/(10**data_y)).mean(axis =1)
# average over realizations
pred_y_rs= pred_y.reshape((len(emu.scale_bin_centers),5,7, -1), order = 'F')[:,0,:,:]
data_y_rs= data_y.reshape((len(emu.scale_bin_centers),5,7, -1), order = 'F').mean(axis = 1)

R = (10**pred_y_rs - 10**data_y_rs).reshape((18,-1), order = 'F')
cov = R.dot(R.T)/(R.shape[1]-1)
print k
print 'Yerr', np.sqrt(np.diag(cov))/(10**data_y.mean(axis=1))
print '*'*10

np.save(save_fname, cov)


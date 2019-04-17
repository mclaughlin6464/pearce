import numpy as np
from load_ds14 import load_ds14
import h5py
from time import time, sleep
from itertools import product
ds14a_part = load_ds14('/scratch/users/swmclau2/Darksky/ds14_a_1.0000', 10)
#particles = ds14a_part.getsubbox([0,0,0], 10, pad=0, fields=['x','y', 'z'])

# TODO should fail gracefully if memory is exceeded or if p is too small.
downsample_factor = 1e-2

f = h5py.File('/scratch/users/swmclau2/Darksky/ds14_a_1.0000_%.3f_downsample.hdf5'%downsample_factor)#, 'w')

try:
    grp = f.create_group("particles")
except ValueError:
    grp = f['particles']
f.close()
#print grp.keys()

R0 = 5.81342950e6
h = 0.6880620000000001
L = 8000 #Mpc

n_subboxes = 8 # per side

subbox_L = L*1.0/n_subboxes

np.random.seed(int(time())) # TODO pass in seed?
t0 = time()
level = 10
#for idx, (subbox_idx, _subbox) in enumerate(ds14a_part.itersubbox(level=10, pad=1.0, fields=['x','y','z'], return_index = True)):
start_idx = (1, 123, 472) 
_subbox = ds14a_part.getsubbox(start_idx, level=level, pad=1.0, fields=['x','y','z'])

print 'Done'

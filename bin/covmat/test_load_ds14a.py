import numpy as np
from load_ds14 import load_ds14
import h5py
from time import time, sleep
from itertools import product
ds14a_part = load_ds14('/scratch/users/swmclau2/Darksky/ds14_a_1.0000', 10)
#particles = ds14a_part.getsubbox([0,0,0], 10, pad=0, fields=['x','y', 'z'])

# TODO should fail gracefully if memory is exceeded or if p is too small.
downsample_factor = 1e-2

f = h5py.File('/scratch/users/swmclau2/Darksky/ds14_a_1.0000_%.3f_downsample.hdf5'%downsample_factor, 'w')

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
start_idx = (2, 217, 640) 
for idx in product(xrange(2**level), repeat=3):
    if idx[0] < start_idx[0] or (idx[0]== start_idx[0] and idx[1] < start_idx[1]):
        continue # skip these other idxs 
    _subbox = ds14a_part.getsubbox(idx, level=level, pad=1.0, fields=['x','y','z'])
    if len(_subbox) == 0: #empty
        print idx, 'DONE'
        break #done!
    subbox_unmod = _subbox.view(np.float64).reshape(_subbox.shape+(-1,)) 
    subbox = (subbox_unmod + R0)*h/1000

    if idx%5000 == 0:
        print idx,subbox_idx, (time()-t0)/3600.0
        if idx!=0:
            f = h5py.File('/scratch/users/swmclau2/Darksky/ds14_a_1.0000_%.3f_downsample.hdf5'%downsample_factor)

#            all_particles[all_particles<0] = 0.0 # numerical errors
#            all_particles[all_particles>L] = L # numerical errors
            idxs = np.floor_divide(all_particles, subbox_L).astype(int)

            #account for these at the index level instead
            idxs[idxs == -1] = 0
            idxs[idxs== n_subboxes] == n_subboxes-1

            unique_subboxes = np.vstack({tuple(row) for row in idxs})

            grp = f['particles']

            for us in unique_subboxes:
                x_in_sb = all_particles[np.all(idxs == us, axis =1)]
                sb_key = 'subbox_%d%d%d'%tuple(us)

                if sb_key in grp.keys():
                    dset = grp[sb_key]
                    dset.resize( (dset.shape[0] + x_in_sb.shape[0], 3))
                    dset[-x_in_sb.shape[0]:] = x_in_sb
                else:
                    dset = grp.create_dataset(sb_key,  data = x_in_sb, maxshape = (None, 3),  compression = 'gzip')

            f.close()

        all_particles = np.array([], dtype='float32')

    downsample_idxs = np.random.choice(subbox.shape[0], size=int(subbox.shape[0] * downsample_factor))
    particles = subbox[downsample_idxs]
    # edge case
    if particles.shape[0] == 0:
        continue

    all_particles = np.resize(all_particles, (all_particles.shape[0] + particles.shape[0], 3 ))
    all_particles[-particles.shape[0]:, ] = particles
    idx+=1


print 'Done'

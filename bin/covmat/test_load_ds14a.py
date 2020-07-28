import numpy as np
from load_ds14 import load_ds14
import h5py
from time import time, sleep
from itertools import product
import sys

#ds14a_part = load_ds14('/scratch/users/swmclau2/Darksky/ds14_a_1.0000', 10)
ds14a_part = load_ds14('/oak/stanford/orgs/kipac/users/swmclau2/Darksky/ds14_a_1.0000', 10)
#particles = ds14a_part.getsubbox([0,0,0], 10, pad=0, fields=['x','y', 'z'])

# TODO should fail gracefully if memory is exceeded or if p is too small.
downsample_factor = 1e-2

#########################################################################vvvvvvvvvvvvvvvvvvvvvvDELETE Wvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
f = h5py.File('/scratch/users/swmclau2/Darksky/ds14_a_1.0000_%.3f_downsample_v4.hdf5'%downsample_factor, 'w')

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
level = 4 
#for idx, (subbox_idx, _subbox) in enumerate(ds14a_part.itersubbox(level=10, pad=1.0, fields=['x','y','z'], return_index = True)):
start_subbox_idx = (0,0,0)
start_idx = 0
all_particles = np.array([], dtype='float32')

for idx, subbox_idx in enumerate(product(xrange(2**level), repeat=3)):
    if subbox_idx[0] < start_subbox_idx[0] or (subbox_idx[0]== start_subbox_idx[0] and subbox_idx[1] < start_subbox_idx[1]):
        start_idx+=1
        continue # skip these other idxs 
    loop_t0 = time()
    print 'A',
    sys.stdout.flush()
    _subbox = ds14a_part._loadsubbox(subbox_idx, level=level, pad=np.ones((3,)),\
                                    fields=['x','y','z'], cut=None, cut_fields=None)

    if len(_subbox) == 0: #empty
        print idx, subbox_idx, 'DONE'
        break #done!
    subbox_unmod = _subbox.view(np.float64).reshape(_subbox.shape+(-1,)) 
    subbox = (subbox_unmod + R0)*h/1000

    #print idx, (time()-t0)/3600.0
    #print subbox.min(axis=0), subbox.max(axis=0)
    #print '*'*10
    #print 'B', time()-loop_t0
    print 'B',
    if idx%100 == 0:
        print
        print idx,subbox_idx, (time()-t0)/3600.0
        print all_particles.shape
        if idx!=0 and idx!=start_idx+1:
            #print 'AA', time()-loop_t0
            f = h5py.File('/scratch/users/swmclau2/Darksky/ds14_a_1.0000_%.3f_downsample_v4.hdf5'%downsample_factor)
#            all_particles[all_particles<0] = 0.0 # numerical errors
#            all_particles[all_particles>L] = L # numerical errors
            idxs = np.floor_divide(all_particles, subbox_L).astype(int)

            #account for these at the index level instead
            idxs[idxs == -1] = 0
            idxs[idxs== n_subboxes] == n_subboxes-1

            unique_subboxes = np.vstack({tuple(row) for row in idxs})

            grp = f['particles']
            #print 'AB', time()-loop_t0

            for us in unique_subboxes:
                x_in_sb = all_particles[np.all(idxs == us, axis =1)]
                sb_key = 'subbox_%d%d%d'%tuple(us)
                print sb_key, x_in_sb.min(axis=0), x_in_sb.max(axis=0)
                if sb_key in grp.keys():
                    dset = grp[sb_key]
                    dset.resize( (dset.shape[0] + x_in_sb.shape[0], 3))
                    dset[-x_in_sb.shape[0]:] = x_in_sb
                else:
                    dset = grp.create_dataset(sb_key,  data = x_in_sb, maxshape = (None, 3),  compression = 'gzip')

            #print 'AC', time()-loop_t0
            f.close()
            print '*-'*20
            print 
            

        all_particles = np.array([], dtype='float32')

    downsample_idxs = np.random.choice(subbox.shape[0], size=int(subbox.shape[0] * downsample_factor))
    particles = subbox[downsample_idxs]
    print 'C',
    #print 'C', time()-loop_t0
    # edge case
    if particles.shape[0] == 0:
        continue

    
    all_particles = np.resize(all_particles, (all_particles.shape[0] + particles.shape[0], 3 ))
    all_particles[-particles.shape[0]:, ] = particles
    print 'D'
    sys.stdout.flush()
    #print 'D', time()-loop_t0
    #print '*'*20

print 'Done'

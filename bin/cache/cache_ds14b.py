from os import path
import numpy as np
from AbundanceMatching import *
from halotools.sim_manager import RockstarHlistReader, CachedHaloCatalog

#halo_dir = '/nfs/slac/des/fs1/g/sims/yymao/ds14_b_sub/hlists/'
halo_dir = '/scratch/users/swmclau2/hlists/ds_14_b_sub/hlists/'
a = 0.65
z = 1.0/a - 1 # ~ 0.55
fname = path.join(halo_dir,  'hlist_%.5f.list'%a)

columns_to_keep = {'halo_id': (1, 'i8'), 'halo_upid':(6,'i8'), 'halo_mvir':(10, 'f4'), 'halo_x':(17, 'f4'),'halo_y':(18,'f4'), 'halo_z':(19,'f4'),'halo_vx':(20,'f4'), 'halo_vy':(21, 'f4'), 'halo_vz':(22,'f4'),'halo_rvir': (11, 'f4'),'halo_rs':(12,'f4'), 'halo_mpeak':(58, 'f4'),'halo_vmax@mpeak':(72, 'f4')}

simname = 'ds_14_b_sub'


reader = RockstarHlistReader(fname, columns_to_keep, '/scratch/users/swmclau2/halocats/hlist_%.2f.list.%s.hdf5'%(a, simname),\
                             simname,'rockstar', z, 'default', 1000.0, 2.44e9, overwrite=True, header_char = '#')
reader.read_halocat(['halo_rvir', 'halo_rs'], write_to_disk=False, update_cache_log=False)

reader.add_supplementary_halocat_columns()
reader.write_to_disk()
reader.update_cache_log()

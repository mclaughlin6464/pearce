from pearce.mocks import cat_dict
from halotools.sim_manager import CachedHaloCatalog , HaloTableCache
import numpy as np
from os import path

a = 0.645161
z = 1.0/a - 1.0

cache = HaloTableCache()

for boxno in xrange(0,1):
    for reaz in xrange(5):
        print boxno, reaz
        cosmo_params = {'simname':'testbox', 'boxno':boxno, 'realization':reaz, 'scale_factors':[a], 'system':'sherlock'}
        cat = cat_dict[cosmo_params['simname']](**cosmo_params)#construct the specified catalog!

        print cat.version_name

        try:
            cat.load_catalog(a, particles = False, downsample_factor = 1e-2)
        except: #delete the other, try again
            fname = '/scratch/users/swmclau2/halocats/hlist_%0.2f.list.testbox_%02d.hdf5'%(a, boxno) 
            #print cat.version_name
            #print fname
            #print "%0.4f"%a
            cache.remove_entry_from_cache_log("testbox", "rockstar", cat.version_name, "%0.4f"%z\
            , fname, raise_non_existence_exception = False)

            cat.load_catalog(a, particles = False, downsample_factor = 1e-2)
           

        cat.halocat.halo_table.write('/home/users/swmclau2/scratch/LSD_boxes/Testbox_%02d_%02d_LSD.hdf5'%(boxno, reaz), format = 'hdf5', path = 'Trainbox_%02d_%02d_LSD.hdf5'%(boxno,reaz), overwrite = True)

from halotools.sim_manager import PtclTableCache, HaloTableCache, CachedHaloCatalog
from glob import glob
import h5py
import numpy as np

cache = HaloTableCache()
#cache = PtclTableCache()

#simname = 'trainingbox'
simname = 'testbox'
#version_name = 'most_recent_%02d_%01d'
version_name = 'most_recent_%02d'
ptcl_vname = version_name + '_particle_%.2f' % (-1 * np.log10(1e-2))

all_halo_boxes = sorted(glob('/scratch/users/swmclau2/hlists/hlist_1.00.list.trainingbox_??.hdf5'))
#all_boxes = []
#all_ptcl_boxes = sorted(glob('/scratch/users/swmclau2/hlists/ptcl_1.00.list.testbox_most_recent_*_*.hdf5'))
all_boxes = all_halo_boxes#zip(sorted(all_halo_boxes), sorted(all_ptcl_boxes))

print len(all_boxes)
#log = cache.matching_log_entry_generator(simname = simname,
#        redshift = 0.0, dz_tol = 0.1)#[0]
        
#k = 0
#for l in log:
#    print l
#    k+=1

#print k

for boxno, t in enumerate(all_boxes):
    print boxno, t
    #print t
    #boxno1, boxno2 = boxno%5, int(boxno/5)
    #print boxno1, boxno2
    with h5py.File(t) as f:
        f.attrs['fname'] = t
    #print ptcl_vname%(boxno2, boxno1)
    try:
        log = list(cache.matching_log_entry_generator(simname = simname,
            version_name = version_name%(boxno), redshift = 0.0, dz_tol = 0.1))[0]

    except IndexError: # cache was screwed up, add it back
        log = cache.determine_log_entry_from_fname(t)
    #print log
    cache.update_cached_file_location(t, log.fname)
    #cache.remove_entry_from_cache_log(log.simname,'rockstar', version_name%boxno, 0.0, '/scratch/users/swmclau2/hlists/hlist_1.00.list.trainingbox_%02d.hdf5'%boxno,
    # raise_non_existence_exception=True)#, delete_corresponding_halo_catalog=False)
    cache.remove_entry_from_cache_log(log.simname,'rockstar',version_name%boxno, 0.0, '/scratch/users/swmclau2/halocats/hlist_1.00.list.trainingbox_%02d.hdf5'%boxno, 
     raise_non_existence_exception=True)#, delete_corresponding_halo_catalog=False)
   
    cache.add_entry_to_cache_log(log)

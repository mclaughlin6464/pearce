from halotools.sim_manager import PtclTableCache, HaloTableCache, CachedHaloCatalog
from glob import glob
import h5py
import numpy as np

cache = HaloTableCache()
#cache = PtclTableCache()

simname = 'trainingbox'
version_name = 'most_recent_%02d'
#ptcl_vname = version_name + '_particle_%.2f' % (-1 * np.log10(1e-2))

all_halo_boxes = sorted(glob('/scratch/users/swmclau2/hlists/hlist_1.00.list.trainingbox_??.hdf5'))
#all_boxes = []
#all_ptcl_boxes = sorted(glob('/scratch/users/swmclau2/hlists/ptcl_1.00.list.trainingbox_most_recent_*.hdf5'))

all_boxes = all_halo_boxes#zip(sorted(all_halo_boxes), sorted(all_ptcl_boxes))

print len(all_boxes)

for boxno, t in enumerate(all_boxes):
    print boxno, t

    with h5py.File(t) as f:
        f.attrs['fname'] = t

    log = list(cache.matching_log_entry_generator(simname = simname,
            version_name = version_name%boxno, redshift = 0.0, dz_tol = 0.1))[0]
    #halocat = CachedHaloCatalog(fname = t)
    #halocat.halo_table.write('/home/users/swmclau2/scratch/LSD_boxes/Trainbox_%02d_LSD.hdf5'%boxno, format = 'hdf5', path = 'Trainbox_%02d_LSD.hdf5'%boxno, overwrite = True)
    cache.update_cached_file_location(t, log.fname)
    #cache.remove_entry_from_cache_log(log.simname, log.halo_finder,
    #log.version_name, log.redshift, log.fname, raise_non_existence_exception=False, delete_corresponding_halo_catalog=True)
    #cache.add_entry_to_cache_log(log)

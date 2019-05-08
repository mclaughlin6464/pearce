from halotools.sim_manager import HaloTableCache, CachedHaloCatalog
from glob import glob

cache = HaloTableCache()

all_tboxes = glob('/u/ki/swmclau2/des/halocats/*testbox*.hdf5')

for boxno, t in enumerate(sorted(all_tboxes)):
    print t,
    log = cache.determine_log_entry_from_fname(t)
    print log.version_name
    #halocat = CachedHaloCatalog(fname = t)
    #halocat.halo_table.write('/home/users/swmclau2/scratch/LSD_boxes/Trainbox_%02d_LSD.hdf5'%boxno, format = 'hdf5', path = 'Trainbox_%02d_LSD.hdf5'%boxno, overwrite = True)

    cache.remove_entry_from_cache_log(log.simname, log.halo_finder,
    log.version_name, log.redshift, log.fname, raise_non_existence_exception=False, delete_corresponding_halo_catalog=True)
    #cache.add_entry_to_cache_log(log)

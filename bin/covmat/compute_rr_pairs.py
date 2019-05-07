from halotools.mock_observables.pair_counters import npairs_jackknife_3d
from halotools.mock_observables.catalog_analysis_helpers import cuboid_subvolume_labels

 
import numpy as np

config_fname = '/home/users/swmclau2/Git/pearce/bin/trainer/xi_cosmo_trainer.yaml'

with open(config_fname, 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

min_ptcl = int(cfg['HOD']['min_ptcl']) 
r_bins = np.array(cfg['observation']['bins'] ).astype(float)

def compute_RR(cat, rbins, n_rands= 5, n_sub = 5, n_cores = 16):

    n_cores = cat._check_cores(n_cores)
#pos_m = return_xyz_formatted_array(x_m, y_m, z_m, period=cat.Lbox)
    rbins = np.array(rbins)

    randoms = np.random.random((cat.halocat.ptcl_table['x'].shape[0] * nr, 3)) * cat.Lbox / cat.h  # Solution to NaNs: Just fuck me up with randoms
    print randoms.shape
    j_index_randoms, N_sub_vol = cuboid_subvolume_labels(randoms, n_sub, cat.Lbox/cat.h)
    print j_index_randoms.shape 
    print N_sub_vol
    RR = npairs_jackknife_3d(randoms, randoms, rbins, period=cat.Lbox/cat.h,
                jtags1=j_index_randoms, jtags2=j_index_randoms,
                            N_samples=N_sub_vol, num_threads=n_cores)
    RR = np.diff(RR, axis=1)

    return RR 


cat = TestBox(boxno = 0, realization = 0, system = 'sherlock')
cat.load(1.0, HOD = str('zheng07'), particles = True, downsample_factor = 1e-2)
RR = compute_RR(cat, r_bins, n_rands = 1)
np.savetxt('RR.npy', RR)




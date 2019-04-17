from halotools.mock_observables.pair_counters import npairs_jackknife_3d
from halotools.mock_observables.catalog_analysis_helpers import cuboid_subvolume_labels

 
import numpy as np

config_fname = '/home/users/swmclau2/Git/pearce/bin/trainer/xi_cosmo_trainer.yaml'

with open(config_fname, 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

min_ptcl = int(cfg['HOD']['min_ptcl']) 
r_bins = np.array(cfg['observation']['bins'] ).astype(float)

from itertools import product
def compute_RR_hack(cat, rbins, n_rands= 5, n_sub = 5, n_split=10, n_cores = 16):

    n_cores = cat._check_cores(n_cores)
#pos_m = return_xyz_formatted_array(x_m, y_m, z_m, period=cat.Lbox)
    rbins = np.array(rbins)
    #np.random.seed(0)
    randoms = np.random.random((int(cat.halocat.ptcl_table['x'].shape[0] * n_rands), 3))*cat.Lbox/cat.h  # Solution to NaNs: Just fuck me up with randoms
    print randoms.shape
    split_randoms = np.array_split(randoms, n_split, axis = 0)
    RR = np.zeros((n_sub**3+1, rbins.shape[0]-1))
    for A, B in product(split_randoms, split_randoms):
        print A.shape, B.shape
        j_index_A, N_sub_vol = cuboid_subvolume_labels(A, n_sub, cat.Lbox/cat.h)
        j_index_B, N_sub_vol = cuboid_subvolume_labels(B, n_sub, cat.Lbox/cat.h)

        AB = npairs_jackknife_3d(A, B, rbins, period=cat.Lbox/cat.h,
                        jtags1=j_index_A, jtags2=j_index_B,
                                                    N_samples=N_sub_vol, num_threads=n_cores)
        AB = np.diff(AB, axis=1)
                
        RR+=AB

    return RR

cat = TestBox(boxno = 0, realization = 0, system = 'sherlock')
cat.load(1.0, HOD = str('zheng07'), particles = True, downsample_factor = 1e-2)
RR = compute_RR_hack(cat, r_bins, n_rands = 10)
np.savetxt('RR.npy', RR)





from pearce.mocks.kittens import TestBox
from pearce.mocks import tpcf_subregions
from halotools.mock_observables import tpcf_jackknife
import numpy as np
from collections import OrderedDict
from time import time
from scipy.optimize import minimize_scalar
import yaml


config_fname = '/home/users/swmclau2/Git/pearce/bin/trainer/xi_cosmo_trainer.yaml'

with open(config_fname, 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

nd = float(cfg['HOD']['fixed_nd'] )
min_ptcl = int(cfg['HOD']['min_ptcl']) 
r_bins = np.array(cfg['observation']['bins'] ).astype(float)

hod_param_ranges =  cfg['HOD']['ordered_params'] 


logMmin_bounds = hod_param_ranges['logMmin']


del hod_param_ranges['logMmin']


def make_LHC(ordered_params, N, seed = None):

    if seed is None:
        seed = int(time())
    np.random.seed(seed)

    points = []
    # by linspacing each parameter and shuffling, I ensure there is only one point in each row, in each dimension.
    for plow, phigh in ordered_params.itervalues():
        point = np.linspace(plow, phigh, num=N)
        np.random.shuffle(point)  # makes the cube random.
        points.append(point)
    return np.stack(points).T


# In[6]:


def add_logMmin(hod_params, cat):

    hod_params['logMmin'] = 13.0 #initial guess
    #cat.populate(hod_params) #may be overkill, but will ensure params are written everywhere
    def func(logMmin, hod_params):
        hod_params.update({'logMmin':logMmin}) 
        return (cat.calc_analytic_nd(hod_params) - nd)**2

    res = minimize_scalar(func, bounds = logMmin_bounds, args = (hod_params,), options = {'maxiter':100}, method = 'Bounded')

    # assuming this doens't fail
    #print 'logMmin', res.x
    hod_params['logMmin'] = res.x


# In[7]:


N = 10
LHC = make_LHC(hod_param_ranges, N)
hod_dicts = [dict(zip(hod_param_ranges.keys(), vals)) for vals in LHC]


# In[13]:


from math import ceil
def compute_full_jk(cat, rbins, rand_scalecut = 1.0 ,  n_rands= [5, 5, 5], n_sub = 3):
#np.random.seed(int(time()))

    n_cores = cat._check_cores(1)

    x_g, y_g, z_g = [cat.model.mock.galaxy_table[c] for c in ['x', 'y', 'z']]
#pos_g = return_xyz_formatted_array(x_g, y_g, z_g, period=cat.Lbox)
    pos_g = np.vstack([x_g, y_g, z_g]).T

    x_m, y_m, z_m = [cat.halocat.ptcl_table[c] for c in ['x', 'y', 'z']]
#pos_m = return_xyz_formatted_array(x_m, y_m, z_m, period=cat.Lbox)
    pos_m = np.vstack([x_m, y_m, z_m]).T

    rbins = np.array(rbins)

    rbins_small,rbins_large = list(rbins[rbins < rand_scalecut]), list(rbins[rbins >= rand_scalecut])
    boundary_idx = np.argmax(rbins>rand_scalecut)
    mid_size_ov2 = int(ceil(len(rbins)/4.0))
    rbins_mid = list(rbins[boundary_idx-mid_size_ov2:boundary_idx+mid_size_ov2]) # take a chunk in the middle to cover the overlap

#rbins_large.insert(0, rbins_small[-1]) # make sure the middle bin is not cut

    cov_gg, cov_gg_gm, cov_gm = np.zeros((rbins.shape[0]-1, rbins.shape[0]-1)),\
                                np.zeros((rbins.shape[0]-1, rbins.shape[0]-1)),\
                                np.zeros((rbins.shape[0]-1, rbins.shape[0]-1))

    for rb,idxs,  nr in zip([rbins_small,rbins_mid, rbins_large],\
                            [(0, len(rbins_small)), (boundary_idx-mid_size_ov2, boundary_idx+mid_size_ov2), len(rbins_small), len(rbins)],\
                            n_rands): #
#for rb, nr in zip([rbins_large], n_rands): #

        randoms = np.random.random((pos_m.shape[0] * nr,
                                    3)) * cat.Lbox / cat.h  # Solution to NaNs: Just fuck me up with randoms
        #out = tpcf_subregions(pos_g / cat.h, randoms, rb, sample2 = pos_m/cat.h, period=cat.Lbox / cat.h,
        #                      num_threads=n_cores, Nsub=n_sub, estimator='Landy-Szalay',\
        #                      do_auto1 = True, do_cross = True, do_auto2 = False)
        a, b, c, cov1, cov2, cov3 = tpcf_jackknife(pos_g / cat.h, randoms, rb, sample2 = pos_m/cat.h, period=cat.Lbox / cat.h,
                              num_threads=n_cores, Nsub=n_sub, estimator='Landy-Szalay')

        
        xi_all = np.hstack(out)

        xi_all_cov = np.cov(xi_all.T, bias = True)*(n_sub-1.0)

    cov_gg[idxs[0]:idxs[1]] = xi_all_cov[:len(rb), :len(rb)]
    cov_gm[idxs[0]:idxs[1]] = xi_all_cov[len(rb):, len(rb):]
    cov_gg_gm[idxs[0]:idxs[1]] = xi_all_cov[:len(rb), :len(rb)]

    cov = np.block([[cov_gg, cov_gg_gm], [cov_gg_gm, cov_gm]])
    return cov
# In[ ]:


cov_mats = np.zeros((7,N, 5, 2*len(r_bins)-2, 2*len(r_bins)-2))
for boxno in xrange(7):
    for realization in xrange(5):
        cat = TestBox(boxno = boxno, realization = realization, system = 'sherlock')
        cat.load(1.0, HOD = str('zheng07'), particles = True, downsample_factor = 1e-2)
        for hod_idx, hod_params in enumerate(hod_dicts):
            add_logMmin(hod_params, cat)
            cat.populate(hod_params)
            print boxno, hod_idx
            mat = compute_full_jk(cat, r_bins)
            cov_mats[boxno, hod_idx, realization] = mat
            break
        break
    break
 

# TODO do the reduction here
np.save('/scratch/users/swmclau2/testbox_covmats.npy', cov_mats)
# In[ ]:





# In[ ]:





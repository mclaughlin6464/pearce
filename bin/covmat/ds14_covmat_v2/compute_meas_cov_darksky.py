from pearce.mocks.kittens import DarkSky 
from pearce.mocks import tpcf
#from halotools.mock_observables import tpcf
from halotools.empirical_models import Zheng07Cens, Zheng07Sats
import numpy as np
from collections import OrderedDict
from time import time
from scipy.optimize import minimize_scalar
import yaml
import sys

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


def add_logMmin(hod_params, cat):

    hod_params['logMmin'] = 13.0 #initial guess
    #cat.populate(hod_params) #may be overkill, but will ensure params are written everywhere
    def func(logMmin, hod_params):
        hod_params.update({'logMmin':logMmin}) 
        return (cat.calc_analytic_nd(hod_params, min_ptcl = min_ptcl) - nd)**2

    res = minimize_scalar(func, bounds = logMmin_bounds, args = (hod_params,), options = {'maxiter':100}, method = 'Bounded')

    # assuming this doens't fail
    #print 'logMmin', res.x
    hod_params['logMmin'] = res.x



def compute_obs(cat, rp_bins, randoms):#, rand_scalecut = 1.0 ,  n_rands= [10, 5, 5], n_sub = 3, RR=RR):
#np.random.seed(int(time()))

    n_cores = cat._check_cores(8)#16)

    x_g, y_g, z_g = [cat.model.mock.galaxy_table[c] for c in ['x', 'y', 'z']]
#pos_g = return_xyz_formatted_array(x_g, y_g, z_g, period=cat.Lbox)
    pos_g = np.vstack([x_g, y_g, z_g]).T
    x_m, y_m, z_m = [cat.halocat.ptcl_table[c] for c in ['x', 'y', 'z']]
    pos_m = np.vstack([x_m, y_m, z_m]).T

    rbins = np.array(rbins)

    randoms = (randoms/np.max(randoms))*cat.Lbox/cat.h

    xi_gg, xi_gm = tpcf(pos_g / cat.h, rbins,\
                          randoms=randoms,  period=None,\
                          num_threads=n_cores, estimator='Landy-Szalay',\
                          do_auto1 = True, do_cross = True, RR_precomputed=RR, NR_precomputed=randoms.shape[0])#, do_auto2 = False)
    
    delta_sigma = cat.calc_ds_analytic(rp_bins, xi = xi_gm, rbins = rbins) 
    return np.r_[xi_gg, delta_sigma] 


config_fname = 'xi_cosmo_trainer.yaml'

RR = np.load('RR.npy')
randoms = np.load('/scratch/users/swmclau2/randoms_gm.npy')

with open(config_fname, 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

nd = float(cfg['HOD']['fixed_nd'] )
min_ptcl = int(cfg['HOD']['min_ptcl']) 
r_bins = np.array(cfg['observation']['bins'] ).astype(float)
rp_bins = np.logspace(-0.9, 1.6, 19)
print r_bins
print rp_bins 

hod_param_ranges =  cfg['HOD']['ordered_params'] 

N = 5 
LHC = make_LHC(hod_param_ranges, N, 16)# 23)
hod_dicts = [dict(zip(hod_param_ranges.keys(), vals)) for vals in LHC]

logMmin_bounds = hod_param_ranges['logMmin']

del hod_param_ranges['logMmin']


obs_vals = np.zeros((N, 2*(len(r_bins)-1)))
#obs_vals = np.load('xi_gg_darksky_obs.npy')
from itertools import product
HOD = (Zheng07Cens, Zheng07Sats)

b1, b2, b3 = sys.argv[1], sys.argv[2], sys.argv[3]
start_subbox = (b1,  b2, b3)

start_idx = 64*int(start_subbox[0])+8*int(start_subbox[1])+int(start_subbox[2])
for subbox_idx, subbox in enumerate(product(''.join([str(i) for i in xrange(8)]), repeat = 3)):
    if subbox_idx < start_idx:
        continue
    cat = DarkSky(int(''.join(subbox)), system = 'sherlock')
    cat.load_model(1.0, HOD = HOD, hod_kwargs = {'modlulate_with_cenocc': True})
    cat.load_catalog_no_cache(1.0, min_ptcl=min_ptcl, particles = True)#, downsample_factor = 1e-2)
    for hod_idx, hod_params in enumerate(hod_dicts):
        print 'HOD: %d'%hod_idx
        add_logMmin(hod_params, cat)
        cat.populate(hod_params, min_ptcl = min_ptcl)
        sys.stdout.flush()
        obs = compute_obs(cat, r_bins, rp_bins, randoms, RR)
        obs_vals[hod_idx] = obs

        np.save('xi_gg_gm_darksky_obs_v8_%s%s%s.npy'%(b1,b2,b3), obs_vals)

    break
        


# coding: utf-8

# I'm gonna port this to a script soon. I need to constrain the RedMagic HOD against nd, f_c, and $\langle N_{gal} | M_{>e14} \rangle$. I don't need to do any populations for this so it should be quick. 

# In[49]:

from pearce.mocks.kittens import cat_dict
import numpy as np
from astropy.cosmology import LambdaCDM
from astropy.io import fits
from scipy.linalg import inv
from os import path
import emcee as mc
from itertools import izip

scale_factors = [0.54780, 0.59260, 0.658, 0.71170, 0.8112]

def lnprior(theta, param_names, param_bounds, *args):
    for p, t in izip(param_names, theta):
        low, high = param_bounds[p]

        if np.isnan(t) or t < low or t > high:
            return -np.inf
    return 0

def lnlike(theta, param_names, param_bounds, obs_vals, invcov, mf, mass_bins, hod_kwargs,cat=cat):
    params = dict(zip(param_names, theta))
    #params['f_c']  = fc#params['f_c']
    
    hod = cat.calc_hod(params, **hod_kwargs)
    nd = np.sum(mf*hod)/((cat.Lbox/cat.h)**3)
    mbc = (mass_bins[1:] + mass_bins[:-1])/2.0
    Nm14 = np.mean(hod[mbc>10**14])
    
    pred_vals = np.array([nd, params['f_c'], Nm14])
    delta = pred_vals - obs_vals
    return -delta.dot(invcov.dot(delta))
    #return -invcov[0,0]*delta[0]**2


# In[61]:

def lnprob(theta, *args):
    """
    The total liklihood for an MCMC. Mostly a generic wrapper for the below functions.
    :param theta:
        Parameters for the proposal
    :param args:
        Arguments to pass into the liklihood
    :return:
        Log Liklihood of theta, a float.
    """
    lp = lnprior(theta, *args)
    if not np.isfinite(lp):
        return -np.inf

    return lp + lnlike(theta, *args)

nwalkers = 200
nsteps = 50000
nburn = 0

savedir = '/u/ki/swmclau2/des/PearceMCMC/'


for zbin_m_1, a in enumerate(reversed(scale_factors)):

    zbin = zbin_m_1+1
    z = 1./a -1
    cosmo_params = {'simname':'chinchilla', 'Lbox':400.0, 'scale_factors':[a]}

    cat = cat_dict[cosmo_params['simname']](**cosmo_params)#construct the specified catalog!

    cat.load(a, tol = 0.01, HOD='hasbRedMagic', particles = False)#, hod_kwargs = {'sec_haloprop_key':'halo_log_nfw_conc'})#, hod_kwargs={'split': 0.5})

    fname = '/u/ki/jderose/public_html/bcc/measurement/y3/3x2pt/buzzard/flock/buzzard-2/tpt_Y3_v0.fits'
    hdulist = fits.open(fname)
    zbin = 1
    z_bins = np.array([0.15, 0.3, 0.45, 0.6, 0.75, 0.9])
    nz_zspec = hdulist[8]
    N= sum(row[2+zbin] for row in nz_zspec.data)

    area = 5063 #sq degrees
    full_sky = 41253 #sq degrees

    buzzard = LambdaCDM(H0=70, Om0=0.286, Ode0=0.714, Tcmb0=2.725, Neff=3.04)
    #volIn, volOut = buzzard.comoving_volume(z_bins[zbin-1]), buzzard.comoving_volume(z_bins[zbin])
    volIn, volOut = buzzard.comoving_volume(z_bins[zbin-1]), buzzard.comoving_volume(z_bins[zbin])

    fullsky_volume = volOut-volIn
    survey_volume = fullsky_volume*area/full_sky
    nd = N/survey_volume
    nd_std = np.sqrt(N)/survey_volume

    fc = 0.2
    fc_std = 0.1

    Nm14 = 4
    Nmstd = 2

    obs_vals = np.array([nd.value, fc, Nm14])

    param_bounds = {'logMmin': [11.0, 14.0], 'sigma_logM': [0.01, 1.0], 'logM0': [12.0, 16.0], 'logM1': [12.0, 16.0], 'alpha': [0.8, 1.5], 'f_c': [0.01, 0.9], 'mean_occupation_centrals_assembias_param1':[-1.0,1.0],
    'mean_occupation_satellites_assembias_param1':[-1.0,1.0]}
    param_names = param_bounds.keys()

    cov = np.diag(np.array([nd_std.value**2, fc_std**2, Nmstd**2]) )
    invcov = inv(cov)

    hod_kwargs = {'mass_bin_range': (9,16),
    'mass_bin_size': 0.01}
    #'min_ptcl': 200}
    mf = cat.calc_mf(**hod_kwargs)
    mass_bins = np.logspace(hod_kwargs['mass_bin_range'][0],                        hod_kwargs['mass_bin_range'][1],                        int( (hod_kwargs['mass_bin_range'][1]-hod_kwargs['mass_bin_range'][0])/hod_kwargs['mass_bin_size'] )+1 )


    chain_fname = path.join(savedir,'%d_walkers_%d_steps_chain_redmagic_bias_z%.2f.npy'%(nwalkers, nsteps, z))

    with open(chain_fname, 'w') as f:
        f.write('#' + '\t'.join(param_names)+'\n')


    # In[ ]:

    ncores = 8 
    num_params = len(param_names)

    pos0 = np.zeros((nwalkers, num_params))

    for i, (pname, (plow, phigh)) in enumerate(param_bounds.iteritems()):
        pos0[:,i] = (phigh+plow)/2.0 + np.random.randn(nwalkers)*(phigh-plow)/2.0


    sampler = mc.EnsembleSampler(nwalkers, num_params, lnprob,
                                 threads=ncores, args=(param_names, param_bounds, obs_vals, invcov, mf, mass_bins, hod_kwargs))#, cat))


    for result in sampler.sample(pos0, iterations=nsteps, storechain=False):
        with open(chain_fname, 'a') as f:
            np.savetxt(f, result[0])


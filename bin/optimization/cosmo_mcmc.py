import emcee as mc
from pearce.emulator import OriginalRecipe, ExtraCrispy
from pearce.mocks import cat_dict
import numpy as np
from os import path
training_file = '/u/ki/swmclau2/des/xi_cosmo_trainer/PearceRedMagicXiCosmoFixedNd.hdf5'

a = 1.0
z = 1./a-1.0

fixed_params = {'z':z}#, 'r': 24.06822623}

n_leaves, n_overlap = 1000, 1

em_method = 'gp'
#emu = OriginalRecipe(training_file, method = em_method, fixed_params=fixed_params, downsample_factor = 1.0)

emu = ExtraCrispy(training_file, n_leaves, n_overlap, split_method='random', method = em_method, fixed_params=fixed_params,
                             custom_mean_function = 'linear', downsample_factor = 0.5)

def nll(p):
    # Update the kernel parameters and compute the likelihood.
    # params are log(a) and log(m)
    #ll = 0
    #for emulator, _y in izip(self._emulators, self.y):
    #    emulator.kernel[:] = p
    #    ll += emulator.lnlikelihood(_y, quiet=True)
    emu._emulator.kernel[ab_param_idxs] = p
    #print p
    ll= emu._emulator.lnlikelihood(emu.y, quiet=False)

    # The scipy optimizer doesn't play well with infinities.
    return -ll if np.isfinite(ll) else 1e25

def lnprior(theta, *args):
    return -np.inf if np.any(np.logical_or(theta < -16, theta > 16)) else 0

def lnprob(theta, *args):
    lp = lnprior(theta, *args)
    #print lp
    if not np.isfinite(lp):
        return -np.inf
    output = lp - nll(theta, *args)
    #print output
    return output
    #return lp - nll(theta, *args)


# In[35]:

nwalkers = 100 
nsteps = 500
nburn = 0
param_names = emu.get_param_names()
num_params = len(param_names)
for pn in param_names:
    bounds = emu.get_param_bounds(pn)
    pos0 = np.random.randn(nwalkers, num_params)*(bounds[1]-bounds[0])/6.0 + (bounds[0]+bounds[1])/2.0
ncores = 8 

savedir = '/u/ki/swmclau2/des/PearceMCMC/'
chain_fname = path.join(savedir, '%d_walkers_%d_steps_cosmo_hyperparams_fixed_z_ec.npy'%(nwalkers, nsteps))

with open(chain_fname, 'w') as f:
    f.write('#' + '\t'.join(ab_param_names)+'\n')

sampler = mc.EnsembleSampler(nwalkers, num_params, lnprob, threads=ncores)
for result in sampler.sample(pos0, iterations = nsteps, storechain=False):

    with open(chain_fname, 'a') as f:
        np.savetxt(f, result[0])


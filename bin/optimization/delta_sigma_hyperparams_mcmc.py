import emcee as mc
from pearce.emulator import OriginalRecipe, ExtraCrispy
from pearce.mocks import cat_dict
import numpy as np
from os import path
training_file = '/u/ki/swmclau2/des/ds_trainer2/PearceRedMagicChindhillaDS.hdf5'
a = 0.8112 
z = 1./a-1.0

fixed_params = {'z':z}#, 'r':0.18477483}
#n_leaves, n_overlap = 10, 2
em_method = 'gp'
emu = OriginalRecipe(training_file, method = em_method, fixed_params=fixed_params, downsample_factor = 0.2)

def nll(p):
    # Update the kernel parameters and compute the likelihood.
    # params are log(a) and log(m)
    #ll = 0
    #for emulator, _y in izip(self._emulators, self.y):
    #    emulator.kernel[:] = p
    #    ll += emulator.lnlikelihood(_y, quiet=True)
    #emu._emulator.kernel[:] = p
    emu._emulator.set_parameter_vector(p)
    #print p
    ll= emu._emulator.lnlikelihood(emu.downsample_y, quiet=False)

    # The scipy optimizer doesn't play well with infinities.
    return -ll if np.isfinite(ll) else 1e25


def lnprior(theta, *args):
    return -np.inf if np.any(np.logical_or(theta < -6, theta > 6)) else 0

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

nwalkers = 200 
nsteps = 1000
nburn = 0
pnames = emu.get_param_names()
num_params = 1*(len(pnames)+1)+1
pos0 = np.random.randn(nwalkers, num_params)*2.0
ncores = 16 


savedir = '/u/ki/swmclau2/des/PearceMCMC/'
chain_fname = path.join(savedir, '%d_walkers_%d_steps_delta_sigma_matern32_hyperparams.npy'%(nwalkers, nsteps))

with open(chain_fname, 'w') as f:
    f.write('#' + '\t'.join(pnames)+'\n')

sampler = mc.EnsembleSampler(nwalkers, num_params, lnprob, threads=ncores)
for result in sampler.sample(pos0, iterations = nsteps, storechain=False):

    with open(chain_fname, 'a') as f:
        np.savetxt(f, result[0])


import emcee as mc
from pearce.emulator import OriginalRecipe, ExtraCrispy
from pearce.mocks import cat_dict
import numpy as np
from os import path
training_dir = '/u/ki/swmclau2/des/PearceLHC_wp_z_fscab_emulator/'
a = 1.0
z = 1./a-1.0

fixed_params = {'z':z}#, 'r':0.18477483}
n_leaves, n_overlap = 10, 2
em_method = 'gp'
emu = OriginalRecipe(training_dir, method = em_method, fixed_params=fixed_params)

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

ab_param_names = ['mean_occupation_centrals_assembias_param1',
'mean_occupation_centrals_assembias_slope1',
'mean_occupation_centrals_assembias_split1',
'mean_occupation_satellites_assembias_param1',
'mean_occupation_satellites_assembias_slope1',
'mean_occupation_satellites_assembias_split1']



ab_param_idxs = []
for apn in ab_param_names:
    ab_param_idxs.append(emu._ordered_params.keys().index(apn)+1)
    
ab_param_idxs = np.array(ab_param_idxs)




# In[31]:

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

nwalkers = 100 
nsteps = 100
nburn = 0
num_params = len(ab_param_names)
pos0 = np.random.randn(nwalkers, num_params)*2.0
ncores = 16 


# In[18]:

_x, _y, _yerr = emu.x, emu.y, emu.yerr


# In[51]:

n = 100
emu.x = _x[:n*emu.scale_bin_centers.shape[0], :]
emu.y = _y[:n*emu.scale_bin_centers.shape[0]]
emu.yerr = _yerr[:n*emu.scale_bin_centers.shape[0]]

emu._build_gp(emu._get_initial_guess(None))

savedir = '/u/ki/swmclau2/des/PearceMCMC/'
chain_fname = path.join(savedir, '%d_walkers_%d_steps_fscab_hyperparams.npy'%(nwalkers, nsteps))

with open(chain_fname, 'w') as f:
    f.write('#' + '\t'.join(ab_param_names)+'\n')

sampler = mc.EnsembleSampler(nwalkers, num_params, lnprob, threads=ncores)
for result in sampler.sample(pos0, iterations = nsteps, storechain=False):

    with open(chain_fname, 'a') as f:
        np.savetxt(f, result[0])


import emcee as mc
from pearce.emulator import OriginalRecipe, ExtraCrispy
from pearce.mocks import cat_dict
import numpy as np
from os import path
training_file = '/home/users/swmclau2/scratch/PearceRedMagicXiCosmo.hdf5'
a = 1.0
z = 1./a-1.0

fixed_params = {'z':z, 'r':24.06822623}
n_leaves, n_overlap = 10, 2
em_method = 'gp'
emu = OriginalRecipe(training_file, method = em_method, fixed_params=fixed_params, downsample_factor = 0.1)

# TODO downsample sampling?
def nll(p):
    emu._emulator.set_parameter_vector(p)

    ll = emu._emulator.lnlikelihood(emu.downsample_y, quiet=True)

    return -ll if np.isfinite(ll) else 1e25

def lnprior(theta):
    return -np.inf if np.any(np.logical_or(theta < -15, theta > 15)) else 0

def lnprob(theta, *args):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf

    return lp - nll(theta, *args)

#p0 = emu._emulator.get_parameter_vector()

#p0 = np.array([  0.       ,  10.6161248,   1.8339794,   7.342365 ,  10.6371797,
     #    2.2441632,  13.8155106,  11.3512804,   3.1795786,   4.6846614,
     #             1.       ,   5.0188608,   3.7658774,  -1.5383083])

p0 = np.array([-12.0550382,   0.1054246,   0.2661017,   5.6407612,   0.2408568,   1.1295944,
   0.3643993,  11.5649985,   4.9071932,   4.7031938,   1.,         11.7621938,
     10.6279446,   0.,         10.6161248,   1.8339794,   7.342365   10.6371797,
        2.2441632,  13.8155106,  11.3512804   3.1795786,   4.6846614   1.,
           5.0188608,   3.7658774,  -1.5383083])

nwalkers = 100 
nsteps = 2000
nburn = 0
num_params = p0.shape[0]#len(emu.get_param_names())+1
pos0 = p0+np.random.randn(nwalkers, num_params)
ncores = 16 


savedir = '/home/users/swmclau2/scratch/'
chain_fname = path.join(savedir, '%d_walkers_%d_steps_cosmo_hyperparams.npy'%(nwalkers, nsteps))

pnames = ['amp']
pnames.extend(emu.get_param_names())
with open(chain_fname, 'w') as f:
    f.write('#' + '\t'.join(pnames)+'\t'+ '\t'.join(pnames)+'\tamp'+'\n')

sampler = mc.EnsembleSampler(nwalkers, num_params, lnprob, threads=ncores)
for result in sampler.sample(pos0, iterations = nsteps, storechain=False):

    with open(chain_fname, 'a') as f:
        np.savetxt(f, result[0])


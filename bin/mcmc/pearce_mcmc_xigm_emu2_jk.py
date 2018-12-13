from pearce.emulator import OriginalRecipe, ExtraCrispy, SpicyBuffalo
from pearce.inference import run_mcmc_iterator
import numpy as np
from os import path
import cPickle as pickle

#training_file = '/u/ki/swmclau2/des/xi_cosmo_trainer/PearceRedMagicXiCosmoFixedNd.hdf5'
training_file = '/scratch/users/swmclau2/xi_gm_cosmo/PearceRedMagicXiGMCosmoFixedNd.hdf5'

em_method = 'gp'
split_method = 'random'

load_fixed_params = {'z':0.0}#, 'HOD': 0}

rmin = 1.0
load_fixed_params['rmin'] = rmin

np.random.seed(0)
emu = SpicyBuffalo(training_file, method = em_method, fixed_params=load_fixed_params, custom_mean_function = 'linear', downsample_factor = 0.1)

fixed_params = {}#'f_c':1.0}#,'logM1': 13.8 }# 'z':0.0}

# NOTE values should be wrong, but it won't matter for the chains.
emulation_point = [('logM0', 13.5), ('sigma_logM', 0.25),
                    ('alpha', 0.9),('logM1', 13.5)]

em_params = dict(emulation_point)
em_params.update(fixed_params)

rpoints = emu.scale_bin_centers

boxno, realization = 0, 1

with open('cosmo_gm_param_dict_%d%d.pkl'%(boxno, realization), 'r') as f:
    cosmo_param_dict = pickle.load(f) 

y = np.loadtxt('xi_gm_true_jk_%d%d.npy'%(boxno, realization))
y = y[-emu.n_bins:]

#emu1_cov = emu.ycov
#shot_cov = np.loadtxt('xi_gg_shot_cov_true.npy')
jk_cov = np.loadtxt('xi_gm_cov_true_jk_%d%d.npy'%(boxno, realization))
sample_cov = np.loadtxt('xigm_scov.npy')

cov = sample_cov + jk_cov
cov = cov[-emu.n_bins:, :][:, -emu.n_bins:]

#em_params.update( cosmo_param_dict)

fixed_params.update(em_params)
#fixed_params.update(cosmo_param_dict)
em_params = cosmo_param_dict

param_names = [k for k in em_params.iterkeys() if k not in fixed_params]

nwalkers = 500 
nsteps = 20000 
nburn = 0 

savedir = '/scratch/users/swmclau2/PearceMCMC/'
#chain_fname = path.join(savedir, '%d_walkers_%d_steps_chain_cosmo_zheng_xi_lowmsat.npy'%(nwalkers, nsteps ))
chain_fname = path.join(savedir, '%d_walkers_%d_steps_xigm_r%d_%d_rmin_emu2_jk.npy'%(nwalkers, nsteps, boxno, realization))

with open(chain_fname, 'w') as f:
    f.write('#' + '\t'.join(param_names)+'\n')

print 'starting mcmc'
np.random.seed(0)
for pos in run_mcmc_iterator([emu], param_names, [y], [cov], rpoints, fixed_params = fixed_params,nwalkers = nwalkers,\
        nsteps = nsteps, nburn = nburn):#, ncores = 1):#, resume_from_previous = chain_fname):

        with open(chain_fname, 'a') as f:
            np.savetxt(f, pos)

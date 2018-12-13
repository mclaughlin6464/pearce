from pearce.emulator import OriginalRecipe, ExtraCrispy, SpicyBuffalo
from pearce.inference import run_mcmc_iterator
import numpy as np
from os import path
import cPickle as pickle

#training_file = '/u/ki/swmclau2/des/xi_cosmo_trainer/PearceRedMagicXiCosmoFixedNd.hdf5'
training_file = '/scratch/users/swmclau2/xi_zheng07_cosmo_lowmsat/PearceRedMagicXiCosmoFixedNd.hdf5'

em_method = 'gp'
split_method = 'random'

load_fixed_params = {'z':0.0}#, 'HOD': 0}

np.random.seed(0)
emu = SpicyBuffalo(training_file, method = em_method, fixed_params=load_fixed_params, custom_mean_function = 'linear', downsample_factor = 0.1)

fixed_params = {}#'f_c':1.0}#,'logM1': 13.8 }# 'z':0.0}

emulation_point = [('logM0', 14.0), ('sigma_logM', 0.2),
                    ('alpha', 1.083),('logM1', 13.7)]#, ('logMmin', 12.233)]

em_params = dict(emulation_point)
em_params.update(fixed_params)

with open('cosmo_param_dict.pkl', 'r') as f:
    cosmo_param_dict = pickle.load(f) 

y = np.loadtxt('xi_gg_true_jk.npy')

emu1_cov = emu.ycov
shot_cov = np.loadtxt('xi_gg_shot_cov_true.npy')
#jk_cov = np.loadtxt('xi_gg_cov_true_jk.npy')
#sample_cov = np.loadtxt('xigg_scov_log.npy')

cov = emu1_cov + shot_cov

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
chain_fname = path.join(savedir, '%d_walkers_%d_steps_xigg_m3_1_lin_emu1_shot.npy'%(nwalkers, nsteps))

with open(chain_fname, 'w') as f:
    f.write('#' + '\t'.join(param_names)+'\n')

print 'starting mcmc'
rpoints = emu.scale_bin_centers
np.random.seed(0)
for pos in run_mcmc_iterator([emu], param_names, [y], [cov], rpoints, fixed_params = fixed_params,nwalkers = nwalkers,\
        nsteps = nsteps):#, nburn = nburn, ncores = 1):#, resume_from_previous = chain_fname):

        with open(chain_fname, 'a') as f:
            np.savetxt(f, pos)

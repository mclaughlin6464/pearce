from pearce.emulator import OriginalRecipe, ExtraCrispy, SpicyBuffalo
from pearce.inference import run_mcmc_iterator
import numpy as np
from os import path
import cPickle as pickle

training_file_gg = '/scratch/users/swmclau2/xi_zheng07_cosmo_lowmsat/PearceRedMagicXiCosmoFixedNd.hdf5'
training_file_gm = '/scratch/users/swmclau2/xi_gm_cosmo/PearceRedMagicXiGMCosmoFixedNd.hdf5'

em_method = 'gp'
split_method = 'random'

load_fixed_params = {'z':0.0}#, 'HOD': 0}
rmin = 1.0 #
load_fixed_params['rmin'] = rmin

np.random.seed(0)
emu_gg = SpicyBuffalo(training_file_gg, method = em_method, fixed_params=load_fixed_params, custom_mean_function = 'linear', downsample_factor = 0.1)
np.random.seed(0)
emu_gm = SpicyBuffalo(training_file_gm, method = em_method, fixed_params=load_fixed_params, custom_mean_function = 'linear', downsample_factor = 0.1)

fixed_params = {}#'f_c':1.0}#,'logM1': 13.8 }# 'z':0.0}

# NOTE values are likely wrong, but thats ok if we're integrating HOD too 
emulation_point = [('logM0', 13.5), ('sigma_logM', 0.25),
                    ('alpha', 0.9),('logM1', 13.5)]#, ('logMmin', 12.233)]

em_params = dict(emulation_point)
em_params.update(fixed_params)

boxno, realization = 3, 0

rpoints = emu_gg.scale_bin_centers

with open('cosmo_param_dict_%d%d.pkl'%(boxno, realization), 'r') as f:
    cosmo_param_dict = pickle.load(f) 

y_gg = np.loadtxt('xi_gg_true_jk_%d%d.npy'%(boxno, realization))
y_gg = y_gg[-emu_gg.n_bins:]

y_gm = np.loadtxt('xi_gm_true_jk_%d%d.npy'%(boxno, realization))
y_gm = y_gm[-emu_gm.n_bins:]

#emu1_cov = emu.ycov
#shot_cov = np.loadtxt('xi_gg_shot_cov_true.npy')
jk_cov = np.loadtxt('xi_gg_cov_true_jk_%d%d.npy'%(boxno, realization))
sample_cov = np.loadtxt('xigg_scov.npy')

cov_gg = sample_cov + jk_cov
cov_gg = cov_gg[-emu_gg.n_bins:, :][:, -emu_gg.n_bins:]

jk_cov = np.loadtxt('xi_gm_cov_true_jk_%d%d.npy'%(boxno, realization))
sample_cov = np.loadtxt('xigm_scov.npy')

cov_gm = sample_cov + jk_cov
cov_gm = cov_gm[-emu_gm.n_bins:, :][:, -emu_gm.n_bins:]

em_params.update( cosmo_param_dict)
#em_params = cosmo_param_dict

param_names = [k for k in em_params.iterkeys() if k not in fixed_params]

nwalkers = 500 
nsteps = 20000 
nburn = 0 

savedir = '/scratch/users/swmclau2/PearceMCMC/'
#chain_fname = path.join(savedir, '%d_walkers_%d_steps_chain_cosmo_zheng_xi_lowmsat.npy'%(nwalkers, nsteps ))
chain_fname = path.join(savedir, '%d_walkers_%d_steps_xi_gg_gm_hod_r%d_%d_rmin_emu2_jk.npy'%(nwalkers, nsteps, boxno, realization))

with open(chain_fname, 'w') as f:
    f.write('#' + '\t'.join(param_names)+'\n')

print 'starting mcmc'
np.random.seed(0)
for pos in run_mcmc_iterator([emu_gg, emu_gm], param_names, [y_gg, y_gm], [cov_gg, cov_gm], rpoints, fixed_params = fixed_params,nwalkers = nwalkers,\
        nsteps = nsteps, nburn = nburn):#, ncores = 1):#, resume_from_previous = chain_fname):

        with open(chain_fname, 'a') as f:
            np.savetxt(f, pos)

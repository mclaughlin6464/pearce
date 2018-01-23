from pearce.emulator import OriginalRecipe, ExtraCrispy
from pearce.mocks import cat_dict
from pearce.inference import run_mcmc_iterator
import numpy as np
from os import path

training_dir = '/u/ki/swmclau2/des/PearceLHC_wt_z/'

em_method = 'gp'
split_method = 'random'

a = 0.81120
#a = 1./(1.0+z)
z = 1.0/a-1.0
load_fixed_params = {'z':z}

emu = ExtraCrispy(training_dir,10, 2, split_method, method=em_method, fixed_params=load_fixed_params)

#Remember if training data is an LHC can't load a fixed set, do that after
#fixed_params = {'f_c':1.0}#,'logM1': 13.8 }# 'z':0.0}

cosmo_params = {'simname':'chinchilla', 'Lbox':400.0, 'scale_factors':[a]}
cat = cat_dict[cosmo_params['simname']](**cosmo_params)#construct the specified catalog!
cat.load(a, HOD='redMagic')
emulation_point = [('f_c', 0.15), ('logM0', 12.0), ('sigma_logM', 0.366), 
                    ('alpha', 1.083),('logM1', 13.7), ('logMmin', 12.733)]


em_params = dict(emulation_point)
fixed_params = {}
#fixed_params = {'mean_occupation_centrals_assembias_param1':0.0,
#                'disp_func_slope_centrals':1.0,
#                'mean_occupation_satellites_assembias_param1':0.0,
#                'disp_func_slope_satellites':1.0}

#em_params.update(fixed_params)
#del em_params['z']
theta_bins = np.logspace(np.log10(2.5), np.log10(250), 21)/60
tpoints = (theta_bins[1:]+theta_bins[:-1])/2

wt_vals = []
nds = []
for i in xrange(5):
    cat.populate(em_params)
    wt_vals.append(cat.calc_wt(theta_bins))
    nds.append(cat.calc_number_density())
#y = np.mean(np.log10(np.array(wp_vals)),axis = 0 )
y = np.loadtxt('buzzard2_wt_11.npy')
# TODO need a way to get a measurement cov for the shams
cov = np.cov(np.array(wt_vals).T)#/np.sqrt(50)
#obs_nd = np.mean(np.array(nds))
obs_nd = np.loadtxt('buzzard2_nd_11.npy')
obs_nd_err = np.std(np.array(nds))*1e4 #TODO delete me

print obs_nd
print cat.calc_analytic_nd(em_params)
print y
print emu.emulate_wrt_r(em_params, tpoints)
print tpoints
print emu.get_param_bounds('r')
#from sys import exit 
#exit(0)

param_names = [k for k in em_params.iterkeys() if k not in fixed_params]

nwalkers = 200
nsteps = 5000
nburn = 0 

print 'Chain starting.'

savedir = '/u/ki/swmclau2/des/PearceMCMC/'
chain_fname = path.join(savedir,'%d_walkers_%d_steps_chain_wt_redmagic_z%.2f_no_nd.npy'%(nwalkers, nsteps, z)) 

with open(chain_fname, 'w') as f:
    f.write('#' + '\t'.join(param_names)+'\n')

for pos in run_mcmc_iterator(emu, cat, param_names, y, cov, tpoints,obs_nd, obs_nd_err,'calc_analytic_nd', fixed_params = fixed_params,nwalkers = nwalkers, nsteps = nsteps, nburn = nburn):#,\
                        #resume_from_previous = '/u/ki/swmclau2/des/PearceMCMC/100_walkers_1000_steps_chain_shuffled_sham.npy')#, ncores = 1)

    with open(chain_fname, 'a') as f:
        np.savetxt(f, pos)

#chain = run_mcmc(emu, cat, param_names, y, cov, tpoints,obs_nd, obs_nd_err,'calc_analytic_nd', fixed_params = fixed_params,\
#        nwalkers = nwalkers, nsteps = nsteps, nburn = nburn)#,\
        #resume_from_previous = '/u/ki/swmclau2/des/PearceMCMC/100_walkers_1000_steps_chain_standard_errors_fixed_points_no_nd.npy')#, ncores = 1)

#savedir = '/u/ki/swmclau2/des/PearceMCMC/'
#np.savetxt(path.join(savedir, '%d_walkers_%d_steps_chain_wt_redmagic_z%.2f.npy'%(nwalkers, nsteps, z)), chain)
#np.savetxt(path.join(savedir, '%d_walkers_%d_steps_truth_ld_errors_2.npy'%(nwalkers, nsteps)),\
#                                np.array([em_params[p] for p in param_names]))
#np.savetxt(path.join(savedir, '%d_walkers_%d_steps_fixed_old_errors_2.npy'%(nwalkers, nsteps)),\
#                                np.array([fixed_params[p] for p in param_names if p in fixed_params]))


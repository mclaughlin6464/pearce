from pearce.emulator import OriginalRecipe, ExtraCrispy
from pearce.mocks import cat_dict
from pearce.inference import run_mcmc_iterator
import numpy as np
from os import path

sherlock = False 
if sherlock:
    training_dir = '/home/swmclau2/scratch/PearceLHC_wt_z/'
else:
    training_dir = '/u/ki/swmclau2/des/PearceLHC_wt_z/'

em_method = 'gp'
split_method = 'random'

a = 0.81120
#a = 1./(1.0+z)
z = 1.0/a-1.0
load_fixed_params = {'z':z}

#emu = ExtraCrispy(training_dir,10, 2, split_method, method=em_method, fixed_params=load_fixed_params)

#Remember if training data is an LHC can't load a fixed set, do that after
#fixed_params = {'f_c':1.0}#,'logM1': 13.8 }# 'z':0.0}

cosmo_params = {'simname':'chinchilla', 'Lbox':400.0, 'scale_factors':[a],'system':'ki-ls'}
cat = cat_dict[cosmo_params['simname']](**cosmo_params)#construct the specified catalog!
#cat.load(a, HOD='hsabRedMagic')
cat.load_catalog(a, tol=0.01, check_sf=False, particles = False)
cat.load_model(a, HOD='hsabRedMagic', check_sf=False)#, hod_kwargs=hod_kwargs)

emulation_point = [('f_c', 0.15), ('logM0', 12.0), ('sigma_logM', 0.266), 
                    ('alpha', 0.9),('logM1', 13.7), ('logMmin', 13.733),
                    ('mean_occupation_satellites_assembias_param1', 0.0), 
                    ('mean_occupation_centrals_assembias_param1', 0.0)]


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
#tpoints = emu.scale_bin_centers

W = 0.00275848072207
rbins = np.array( [0.31622777, 0.44326829, 0.62134575, 0.87096359, 1.22086225, 1.7113283, 2.39883292, 3.36253386, 4.71338954, 6.60693448, 9.26118728,  12.98175275, 18.19700859,  25.50742784,  35.75471605,  50.11872336])

wt_vals = []
nds = []
for i in xrange(25):
    cat.populate(em_params)
    wt_vals.append(cat.calc_wt(theta_bins, rbins = rbins, W=W))
    nds.append(cat.calc_number_density())
#y = np.mean(np.log10(np.array(wp_vals)),axis = 0 )
y = np.loadtxt('buzzard2_wt_11.npy')
# TODO need a way to get a measurement cov for the shams
cov = np.cov(np.array(wt_vals).T)#/np.sqrt(50)
#cov = np.loadtxt('wt_11_cov.npy')
cov = np.cov(np.array(wt_vals).T)#/np.sqrt(50)
np.savetxt('wt_11_cov.npy', cov)

log_cov = np.cov(np.array(np.log10(wt_vals)).T)#/np.sqrt(50)
np.savetxt('wt_11_log_cov.npy', log_cov)

#from sys import exit
#exit(0)
#obs_nd = np.mean(np.array(nds))
obs_nd = np.loadtxt('buzzard2_nd_11.npy')
obs_nd_err = np.std(np.array(nds))

np.savetxt('nd_11_cov.npy', np.array(obs_nd_err))
#obs_nd_err = 1e-3

#print cat.calc_analytic_nd(em_params)
#print cat.calc_number_density(em_params)
from sys import exit 
exit(0)

param_names = [k for k in em_params.iterkeys() if k not in fixed_params]

nwalkers = 200
nsteps = 5000
nburn = 0 

print 'Chain starting.'

if sherlock:
    savedir = '/home/swmclau2/scratch/PearceMCMC/'
else:
    savedir = '/u/ki/swmclau2/des/PearceMCMC/'
chain_fname = path.join(savedir,'%d_walkers_%d_steps_chain_wt_redmagic_z%.2f.npy'%(nwalkers, nsteps, z)) 

with open(chain_fname, 'w') as f:
    f.write('#' + '\t'.join(param_names)+'\n')

for pos in run_mcmc_iterator(emu, cat, param_names, y, cov, tpoints,obs_nd, obs_nd_err,'calc_analytic_nd', fixed_params = fixed_params,nwalkers = nwalkers, nsteps = nsteps, nburn = nburn, ncores=8):#,\
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



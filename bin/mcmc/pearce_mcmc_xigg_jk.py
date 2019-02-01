from pearce.emulator import OriginalRecipe, ExtraCrispy, SpicyBuffalo
from pearce.mocks import cat_dict
from pearce.inference import run_mcmc_iterator
from scipy.optimize import minimize_scalar
import numpy as np
from os import path

#training_file = '/u/ki/swmclau2/des/xi_cosmo_trainer/PearceRedMagicXiCosmoFixedNd.hdf5'
training_file = '/scratch/users/swmclau2/xi_zheng07_cosmo_lowmsat/PearceRedMagicXiCosmoFixedNd.hdf5'

em_method = 'gp'
split_method = 'random'

load_fixed_params = {'z':0.0}#, 'HOD': 0}

np.random.seed(0)
emu = SpicyBuffalo(training_file, method = em_method, fixed_params=load_fixed_params, custom_mean_function = 'linear', downsample_factor = 0.1)

print 'Metric:', emu._emulators[0].get_parameter_vector()

#Remember if training data is an LHC can't load a fixed set, do that after
fixed_params = {}#'f_c':1.0}#,'logM1': 13.8 }# 'z':0.0}

cosmo_params = {'simname':'testbox', 'boxno': 0, 'realization':0, 'scale_factors':[1.0], 'system': 'sherlock'}
cat = cat_dict[cosmo_params['simname']](**cosmo_params)#construct the specified catalog!

cat.load(1.0, HOD='zheng07')

emulation_point = [('logM0', 14.0), ('sigma_logM', 0.2), 
                    ('alpha', 1.083),('logM1', 13.7)]#, ('logMmin', 12.233)]

em_params = dict(emulation_point)
em_params.update(fixed_params)

def add_logMmin(hod_params, cat):
    """
    In the fixed number density case, find the logMmin value that will match the nd given hod_params
    :param: hod_params:
        The other parameters besides logMmin
    :param cat:
        the catalog in question
    :return:
        None. hod_params will have logMmin added to it.
    """
    hod_params['logMmin'] = 13.0 #initial guess
    #cat.populate(hod_params) #may be overkill, but will ensure params are written everywhere
    def func(logMmin, hod_params):
        hod_params.update({'logMmin':logMmin})
        return (cat.calc_analytic_nd(hod_params) - 1e-4)**2

    res = minimize_scalar(func, bounds = (12, 16), args = (hod_params,), options = {'maxiter':100}, method = 'Bounded')

    # assuming this doens't fail
    hod_params['logMmin'] = res.x

add_logMmin(em_params, cat)

r_bins = np.logspace(-1.1, 1.6, 19)
rpoints = emu.scale_bin_centers 

xi_vals = []
for i in xrange(50):
    cat.populate(em_params)
    xi_vals.append(cat.calc_xi(r_bins))

# TODO need a way to get a measurement cov for the shams
xi_vals = np.log10(np.array(xi_vals))

cat.populate(em_params)
y10, cov10 = cat.calc_xi(r_bins, do_jackknife=True, jk_args = {'n_rands':10, 'n_sub':5})
#y10 = np.loadtxt('xi_gg_true_jk.npy')
#cov10 = np.loadtxt('xi_gg_cov_true_jk.npy')

y = np.log10(y10)
#y = np.mean(xi_vals, axis = 0)

shot_cov = np.cov(xi_vals, rowvar = False)

cov = np.log10(1+cov10/(np.outer(y10, y10))) # TODO check this is right?

np.savetxt('xi_gg_true_jk.npy', y)
#np.savetxt('xi_gg_cov_true_jk.npy', cov)

scov = np.loadtxt('xigg_scov_log.npy')

# TODOremoved jackknife
cov+=scov # add the emu covariance

cov+=shot_cov # 

np.savetxt('xi_gg_cov_true_all.npy', cov)

#cov = np.diag(np.diag(cov))

#xi_vals = np.log10(np.array(xi_vals))
#y = np.loadtxt('xi_gg_true.npy') #xi_vals.mean(axis = 0) #take one example as our xi. could also use the mean, but lets not cheat.
#cov = np.loadtxt('xi_gg_cov_true.npy')#/np.sqrt(50)

# get cosmo params
del em_params['logMmin']
cpv = cat._get_cosmo_param_names_vals()

cosmo_param_dict = {key: val for key, val in zip(cpv[0], cpv[1])}

#em_params.update( cosmo_param_dict)

fixed_params.update(em_params)
#fixed_params.update(cosmo_param_dict)
em_params = cosmo_param_dict

param_names = [k for k in em_params.iterkeys() if k not in fixed_params]

nwalkers = 10 
nsteps = 100
nburn = 0 

savedir = '/scratch/users/swmclau2/PearceMCMC/'
#chain_fname = path.join(savedir, '%d_walkers_%d_steps_chain_cosmo_zheng_xi_lowmsat.npy'%(nwalkers, nsteps ))
chain_fname = path.join(savedir, '%d_walkers_%d_steps_chain_cosmo_zheng_xi_jk_meanv2.npy'%(nwalkers, nsteps))

with open(chain_fname, 'w') as f:
    f.write('#' + '\t'.join(param_names)+'\n')

print 'starting mcmc'
np.random.seed(0)
for pos in run_mcmc_iterator([emu], param_names, [y], [cov], rpoints, fixed_params = fixed_params,nwalkers = nwalkers,\
        nsteps = nsteps, nburn = nburn, ncores = 1):#, resume_from_previous = chain_fname):

        with open(chain_fname, 'a') as f:
            np.savetxt(f, pos)

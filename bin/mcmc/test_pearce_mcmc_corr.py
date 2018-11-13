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

metric = {'logM1': [5.5028042800000003, 1.64464882], 'Neff': [12.0, 7.1583221799999999], 'logM0': [10.56121261, 0.79122102000000005], 'sigma_logM': [12.0, 11.9455156], 'H0': [9.0079749899999992, 12.0], 'ln10As': [-8.8855688399999995, 3.3745027799999998], 'alpha': [0.59482347000000002, 4.0302020299999999], 'bias': [2.6860710800000001], 'omch2': [12.0, 0.25342134999999999], 'w0': [-11.18424935, 0.96977813999999996], 'amp': [-4.2960540299999996, -6.2436324499999998], 'ns': [-8.3977249399999998, 12.0], 'ombh2': [-12.0, -8.2664450499999997]}

#metric = dict(zip(metric.keys(), vals))
# TODO this metric input isnt working

#emu = ExtraCrispy(training_dir,10, 2, split_method, method=em_method, fixed_params=load_fixed_params)
np.random.seed(0)
emu = SpicyBuffalo(training_file, method = em_method, fixed_params=load_fixed_params, hyperparams = {'metric': metric},
                         custom_mean_function = 'linear', downsample_factor = 0.1)

print emu._emulators[0].get_parameter_vector()

#v = [ 12. ,         12. ,         12.  ,        12.   ,       12.     ,     12.,
#      11.67920429 ,  6.81656135 , 12.   ,        1.92715272 , 12.    ,       7.72884642,
#      12.,        -12.     ,      2.57697301,  12. ,          8.85016763,
#      9.96558899 ,  6.24704116 , 12.      ,    12.   ,      -12.    ,     -12. ,        -12.,
#      12.    ]

v = [  2.68607108,  -4.29605403, -12.,          12.,         -11.18424935,
          -8.39772494,  -8.88556884,   9.00797499,  12.,          10.56121261,  12.,
             5.50280428,   0.59482347,  -6.24363245,  -8.26644505,   0.25342135,
                0.96977814,  12.,           3.37450278,  12.,           7.15832218,
                   0.79122102,  11.9455156,    1.64464882,   4.03020203]
                                                     

v = np.ones_like(v)*12.0

if hasattr(emu, "_emulator"):
    emu._emulator.set_parameter_vector(v)
    emu._emulator.recompute()
else:
    for _emulator in emu._emulators:
        _emulator.set_parameter_vector(v)
        _emulator.recompute()

#Remember if training data is an LHC can't load a fixed set, do that after
fixed_params = {}#'f_c':1.0}#,'logM1': 13.8 }# 'z':0.0}

cosmo_params = {'simname':'testbox', 'boxno': 3, 'realization':0, 'scale_factors':[1.0], 'system': 'sherlock'}
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

#xi_vals = []
#for i in xrange(50):
#    cat.populate(em_params)
#    xi_vals.append(cat.calc_xi(r_bins))

# TODO need a way to get a measurement cov for the shams
#xi_vals = np.log10(np.array(xi_vals))
y = np.loadtxt('xi_gg_true.npy') #xi_vals.mean(axis = 0) #take one example as our xi. could also use the mean, but lets not cheat.
cov = np.loadtxt('xi_gg_cov_true.npy')#/np.sqrt(50)

# get cosmo params
del em_params['logMmin']
cpv = cat._get_cosmo_param_names_vals()

cosmo_param_dict = {key: val for key, val in zip(cpv[0], cpv[1])}

em_params.update( cosmo_param_dict)

#fixed_params.update(em_params)
#fixed_params.update(cosmo_param_dict)
#em_params = cosmo_param_dict

#last_param = 'omch2'
#em_params = {last_param: fixed_params[last_param]}
#del fixed_params[last_param]

param_names = [k for k in em_params.iterkeys() if k not in fixed_params]

nwalkers = 1000 
nsteps = 50000
nburn = 0 

savedir = '/scratch/users/swmclau2/PearceMCMC/'
#chain_fname = path.join(savedir, '%d_walkers_%d_steps_chain_cosmo_zheng_xi_lowmsat.npy'%(nwalkers, nsteps ))
chain_fname = path.join(savedir, '%d_walkers_%d_steps_chain_cosmo_zheng_xi_lowmsat.npy'%(nwalkers, nsteps))

with open(chain_fname, 'w') as f:
    f.write('#' + '\t'.join(param_names)+'\n')

print 'starting mcmc'
for pos in run_mcmc_iterator([emu], param_names, [y], [cov], rpoints, fixed_params = fixed_params,nwalkers = nwalkers,\
        nsteps = nsteps, nburn = nburn):#, resume_from_previous = chain_fname):

        with open(chain_fname, 'a') as f:
            np.savetxt(f, pos)

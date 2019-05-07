from pearce.emulator import OriginalRecipe, ExtraCrispy, SpicyBuffalo
from pearce.mocks import cat_dict
import numpy as np
from os import path
from george.kernels import *
import GPyOpt

training_file = '/scratch/users/swmclau2/xi_zheng07_cosmo_lowmsat/PearceRedMagicXiCosmoFixedNd.hdf5'
em_method = 'gp'
from sys import argv
ridx = int(argv[1])

rpoints = np.load('rpoints.npy')
fixed_params = {'z':0.0, 'r': rpoints[ridx]}
#emu = SpicyBuffalo(training_file, method = em_method, fixed_params=fixed_params,
#                 custom_mean_function = 'linear', downsample_factor = 0.01)
from time import time
np.random.seed(int(time()))
emu = OriginalRecipe(training_file, method = em_method, fixed_params=fixed_params,
                 custom_mean_function = 'linear', downsample_factor = 0.1)

#emu = ExtraCrispy(training_file, n_leaves, n_overlap, split_method='random', method = em_method, fixed_params=fixed_params,
#                             custom_mean_function = 'linear', downsample_factor = 0.5)

#emu = SpicyBuffalo(training_file, method = em_method, fixed_params=fixed_params,
#                 custom_mean_function = 'linear', downsample_factor = 0.1)
sbc = emu.scale_bin_centers

def nll(p):
    # Update the kernel parameters and compute the likelihood.
    # params are log(a) and log(m)
    y = getattr(emu, "downsample_y", emu.y)


    if hasattr(emu, "_emulator"):
        emu._emulator.set_parameter_vector(p[0])
        ll= emu._emulator.lnlikelihood(y, quiet=False)
    else:

        ll = 0
        for _y, _emu in zip(y, emu._emulators):
            _emu.set_parameter_vector(p[0])
            ll+=_emu.lnlikelihood(_y, quiet = False)

    # The scipy optimizer doesn't play well with infinities.
    return -ll if np.isfinite(ll) else 1e25


param_names = emu.get_param_names()
param_names_2 = ['amp1', 'amp2']
param_names_2.extend(param_names)
param_names_2.append('amp3')
param_names_2.extend(param_names)


#num_params = 2*(1+len(param_names)) + 1
num_params =  len(emu._emulator.get_parameter_vector())

space = [{'name': name, 'type': 'continuous', 'domain': (-12, 12)} for name in param_names_2]

feasible_region = GPyOpt.Design_space(space = space)

max_iter = 1000
tol = 1e-8

initial_design = GPyOpt.experiment_design.initial_design('random', feasible_region, 10)
# --- CHOOSE the objective
objective = GPyOpt.core.task.SingleObjective(nll)

# --- CHOOSE the model type
model = GPyOpt.models.GPModel(exact_feval=True,optimize_restarts=10,verbose=False)

# --- CHOOSE the acquisition optimizer
aquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(feasible_region)

# --- CHOOSE the type of acquisition
acquisition = GPyOpt.acquisitions.AcquisitionEI(model, feasible_region, optimizer=aquisition_optimizer)

# --- CHOOSE a collection method
evaluator = GPyOpt.core.evaluators.Sequential(acquisition)

bo = GPyOpt.methods.ModularBayesianOptimization(model, feasible_region, objective, acquisition, evaluator, initial_design)


bo.run_optimization(max_iter = max_iter, max_time = 24*60*60, eps = tol, verbosity=False) 

print emu._emulator.kernel
print 'Result', bo.x_opt


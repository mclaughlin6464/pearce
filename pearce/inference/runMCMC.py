#!/bin/bash
"""This file hold the function run_mcmc, which  takes a trained emulator and a set of truth data and runs
    and MCMC analysis with a predefined number of steps and walkers."""


from multiprocessing import cpu_count
import warnings

import numpy as np
import emcee as mc
from scipy.linalg import inv

from .defaultProb import lnprob

def run_mcmc(emu,param_names, y, cov, r_bin_centers, fixed_params = {},  nwalkers=1000, nsteps=100, nburn=20, n_cores='all'):

    assert n_cores == 'all' or n_cores > 0
    if type(n_cores) is not str:
        assert int(n_cores) == n_cores

    max_cores = cpu_count()
    if n_cores == 'all':
        n_cores = max_cores
    elif n_cores > max_cores:
        warnings.warn('n_cores invalid. Changing from %d to maximum %d.' % (n_cores, max_cores))
        n_cores = max_cores
        # else, we're good!

    assert y.shape[0] == cov.shape[0] and cov.shape[1] == cov.shape[0]
    assert y.shape[0] == r_bin_centers.shape[0]

    # check we've defined all necessary params
    assert emu.emulator_ndim == len(fixed_params) + len(param_names)
    tmp = param_names[:]
    tmp.extend(fixed_params.keys())
    assert emu.check_param_names(tmp, ignore = ['r'])

    num_params = len(param_names)

    combined_inv_cov = inv(emu.ycov + cov)

    sampler = mc.EnsembleSampler(nwalkers, num_params, lnprob,
                                 threads=n_cores, args=(param_names, emu, r_bin_centers, y, combined_inv_cov))

    pos0 = np.zeros((nwalkers, num_params))
    for idx, pname in enumerate(param_names):
        low, high = emu.get_param_bounds(pname)
        pos0[:,idx] = np.random.randn(size = nwalkers)*(np.abs(high-low)/6.0) + (low+high)/2.0

    sampler.run_mcmc(pos0, nsteps)

    chain = sampler.chain[:, nburn:, :].reshape((-1, num_params))

    return chain
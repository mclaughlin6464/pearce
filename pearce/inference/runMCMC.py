#!/bin/bash
"""This file hold the function run_mcmc, which  takes a trained emulator and a set of truth data and runs
    and MCMC analysis with a predefined number of steps and walkers."""


from multiprocessing import cpu_count
import warnings
from itertools import izip

import numpy as np
import emcee as mc
from scipy.linalg import inv

#from .defaultProb import lnprob
#emu = None

def lnprior(theta, param_names, *args):
    # TODO copy docs over
    #global emu
    for p, t in izip(param_names, theta):
        low, high = emu.get_param_bounds(p)
        if np.isnan(t) or t < low or t > high:
            return -np.inf
    return 0


def lnlike(theta, param_names, r_bin_centers, y, combined_inv_cov):
    # TODO copy docs over
    #global emu
    y_bar = emu.emulate_wrt_r(dict(izip(param_names, theta)), r_bin_centers)[0] 
    # should chi2 be calculated in log or linear?
    # answer: the user is responsible for taking the log before it comes here.
    delta = y_bar - y
    print y
    print y_bar
    return -0.5 * np.dot(delta, np.dot(combined_inv_cov, delta))


def lnprob(theta, *args):

    lp = lnprior(theta, *args)
    if not np.isfinite(lp):
        return -np.inf

    return lp + lnlike(theta, *args)

def run_mcmc(_emu,param_names, y, cov, r_bin_centers, fixed_params = {},  nwalkers=1000, nsteps=100, nburn=20, n_cores='all'):

    emu = _emu
    global emu

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
    assert emu.emulator_ndim == len(fixed_params) + len(param_names) +1 #for r
    tmp = param_names[:]
    tmp.extend(fixed_params.keys())
    assert emu.check_param_names(tmp, ignore = ['r'])

    num_params = len(param_names)

    combined_inv_cov = inv(emu.ycov + cov)

    ### speed up parallel emcee by removing some of the objects the emus don't need to make predictons.
    ### return at the end!
    '''
    if n_cores > 1:
        x_copy, yerr_copy = emu.x, emu.yerr
        del emu.x
        del emu.yerr

        try:
            emulator = emu._emulator
            emulator_x_copy, emulator_yerr_copy = emulator._x, emulator._yerr
            del emulator._x
            del emulator._yerr
        except AttributeError: #has _emulators instead i.e. ExtraCrispy
            emulators =  emu._emulators
            emulator_x_copies, emulator_yerr_copies = [], []
            for e in emulators:
                emulator_x_copies.append(e._x)
                del e._x
                emulator_yerr_copies.append(e._yerr)
                del e._yerr
    '''
    sampler = mc.EnsembleSampler(nwalkers, num_params, lnprob,
                                 threads=n_cores, args=(param_names, r_bin_centers, y, combined_inv_cov))

    pos0 = np.zeros((nwalkers, num_params))
    for idx, pname in enumerate(param_names):
        low, high = emu.get_param_bounds(pname)
        pos0[:,idx] = np.random.randn(nwalkers)*(np.abs(high-low)/6.0) + (low+high)/2.0

    sampler.run_mcmc(pos0, nsteps)

    '''
    if n_cores > 1:
        emu.x = x_copy
        emu.yerr = yerr_copy

        if hasattr(emu, "_emulator"):
            emu._emulator._x = emulator_x_copy
            emu._emulator._yerr = emulator_yerr_copy
        else:
            emulators =  emu._emulators
            for e, exc, eyc in zip(emulators, emulator_x_copies, emulator_yerr_copies):
                e._x  = exc
                e._yerr = eyc
    '''
    chain = sampler.chain[:, nburn:, :].reshape((-1, num_params))

    return chain

#!/bin/bash
"""This file hold the function run_mcmc, which  takes a trained emulator and a set of truth data and runs
    and MCMC analysis with a predefined number of steps and walkers."""


from multiprocessing import cpu_count
import warnings
from itertools import izip

import numpy as np
import emcee as mc
from scipy.linalg import inv

#liklihood functions need to be defined here because the emulator will be made global

def lnprior(theta, param_names, *args):
    """
    Prior for an MCMC. Default is to assume flat prior for all parameters defined by the boundaries the
    emulator is built from. Retuns negative infinity if outside bounds or NaN
    :param theta:
        The parameters proposed by the sampler.
    :param param_names
        The names identifying the values in theta, needed to extract their boundaries
    :return:
        Either 0 or -np.inf, depending if the params are allowed or not.
    """
    for p, t in izip(param_names, theta):
        low, high = _emu.get_param_bounds(p)
        if np.isnan(t) or t < low or t > high:
            return -np.inf
    return 0


def lnlike(theta, param_names, r_bin_centers, y, combined_inv_cov, obs_nd, obs_nd_err, nd_func):
    """
    :param theta:
        Proposed parameters.
    :param param_names:
        The names of the parameters in theta
    :param r_bin_centers:
        The centers of the r bins y is measured in, angular or radial.
    :param y:
        The measured value of the observable to compare to the emulator.
    :param combined_inv_cov:
        The inverse covariance matrix. Explicitly, the inverse of the sum of the mesurement covaraince matrix
        and the matrix from the emulator. Both are independent of emulator parameters, so can be precomputed.
    :param obs_nd
        Observed number density
    :param obs_nd_err
        Uncertainty in the observed nd
    :param nd_func
        Function that can compute the number density given a dictionary of HOD params.
    :return:
        The log liklihood of theta given the measurements and the emulator.
    """
    param_dict = dict(izip(param_names, theta))
    y_bar = _emu.emulate_wrt_r(param_dict, r_bin_centers)[0]
    # should chi2 be calculated in log or linear?
    # answer: the user is responsible for taking the log before it comes here.
    delta = y_bar - y

    chi2 = -0.5 * np.dot(delta, np.dot(combined_inv_cov, delta))

    return chi2 - ((obs_nd-nd_func(param_dict))/obs_nd_err)**2


def lnprob(theta, *args):
    """
    The total liklihood for an MCMC. Mostly a generic wrapper for the below functions.
    :param theta:
        Parameters for the proposal
    :param args:
        Arguments to pass into the liklihood
    :return:
        Log Liklihood of theta, a float.
    """
    lp = lnprior(theta, *args)
    if not np.isfinite(lp):
        return -np.inf

    return lp + lnlike(theta, *args)

def run_mcmc(emu, param_names, y, cov, r_bin_centers, obs_nd, obs_nd_err, nd_func,\
             fixed_params = {},  nwalkers=1000, nsteps=100, nburn=20, ncores='all'):
    """
    Run an MCMC using emcee and the emu. Includes some sanity checks and does some precomputation.
    Also optimized to be more efficient than using emcee naively with the emulator.

    :param emu:
        A trained instance of the Emu object
    :param param_names:
        Names of the parameters to constrain
    :param y:
        data to constrain against
    :param cov:
        measured covariance of y
    :param r_bin_centers:
        The scale bins corresponding to y
    :param obs_nd
        Observed number density
    :param obs_nd_err
        Uncertainty in the observed nd
    :param nd_func
        Function that can compute the number density given a dictionary of HOD params.
    :param fixed_params:
        Any values held fixed during the emulation, default is {}
    :param nwalkers:
        Number of walkers for the mcmc. default is 1000
    :param nsteps:
        Number of steps for the mcmc. Default is 1--
    :param nburn:
        Number of burn in steps, default is 20
    :param ncores:
        Number of cores. Default is 'all', which will use all cores available
    :return:
        chain, collaposed to the shape ((nsteps-nburn)*nwalkers, len(param_names))
    """
    #make emu global so it can be accessed by the liklihood functions
    _emu = emu
    global _emu

    assert ncores == 'all' or ncores > 0
    if type(ncores) is not str:
        assert int(ncores) == ncores

    max_cores = cpu_count()
    if ncores == 'all':
        ncores = max_cores
    elif ncores > max_cores:
        warnings.warn('ncores invalid. Changing from %d to maximum %d.' % (ncores, max_cores))
        ncores = max_cores
        # else, we're good!

    assert y.shape[0] == cov.shape[0] and cov.shape[1] == cov.shape[0]
    assert y.shape[0] == r_bin_centers.shape[0]

    # check we've defined all necessary params
    assert _emu.emulator_ndim == len(fixed_params) + len(param_names) +1 #for r
    tmp = param_names[:]
    tmp.extend(fixed_params.keys())
    assert _emu.check_param_names(tmp, ignore = ['r'])

    num_params = len(param_names)

    combined_inv_cov = inv(_emu.ycov + cov)

    sampler = mc.EnsembleSampler(nwalkers, num_params, lnprob,
                                 threads=ncores, args=(param_names, r_bin_centers, y, combined_inv_cov,\
                                                       obs_nd, obs_nd_err, nd_func))

    pos0 = np.zeros((nwalkers, num_params))
    for idx, pname in enumerate(param_names):
        low, high = _emu.get_param_bounds(pname)
        pos0[:,idx] = np.random.randn(nwalkers)*(np.abs(high-low)/6.0) + (low+high)/2.0

    sampler.run_mcmc(pos0, nsteps)


    chain = sampler.chain[:, nburn:, :].reshape((-1, num_params))

    return chain

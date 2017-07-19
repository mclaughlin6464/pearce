#!/bin/bash
"""This file holds the default liklihood, prior, and probability function for infering the liklihood of model parameters
   using the emulator. """

from time import time
from itertools import izip

import numpy as np

__all__ = ['lnprob','lnprior', 'lnlike']

# These functions cannot be instance methods
# Emcee throws a fit when trying to compile the liklihood functions that are attached
# to the object calling it
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
    out = lp + lnlike(theta, *args)
    return out

# TODO this assumes there's only one emulator. I want this to work for multiple. Not sure the best place/way to do that.
# Don't know if I want each function to only use one emulator or what.
#def lnprior(theta, param_names, emu, *args):
def lnprior(theta, param_names, *args):
    """
    Prior for an MCMC. Default is to assume flat prior for all parameters defined by the boundaries the
    emulator is built from. Retuns negative infinity if outside bounds or NaN
    :param theta:
        The parameters proposed by the sampler.
    :param param_names
        The names identifying the values in theta
    :param emu:
        The emulator object. Needs to be accessed to get the priors.
    :return:
        Either 0 or -np.inf, depending if the params are allowed or not.
    """
    return 0
    '''
    for p, t in izip(param_names, theta):
        low, high = emu.get_param_bounds(p)
        if np.isnan(t) or t < low or t > high:
            return -np.inf
    return 0
    '''

# TODO same as above, this will need tweaks to take multiple emulators.
# TODO is param_names necessary? The reason why is that theta doesn't include possible fixed params or 'r', so
# it is not the same as all the parameters the emulator predicts. This the issue where the emulator predicts over more
# dimensions than we are trying to constrain.
# TODO how to do this with number density?
#def lnlike(theta, param_names, emu, r_bin_centers, y, combined_inv_cov):
def lnlike(theta, param_names,emu, r_bin_centers, y, combined_inv_cov):
    """

n izip(param_names, theta):
        low, high = emu.get_param_bounds(p)
                if np.isnan(t) or t < low or t > high:
                            return -np.inf
                                return 0The liklihood of parameters theta given the other parameters and the emulator.
    :param theta:
        Proposed parameters.
    :param param_names:
        The names of the parameters in theta
    :param emu:
        The emulator object. Used to perform the emulation.
    :param r_bin_centers:
        The centers of the r bins y is measured in, angular or radial.
    :param y:
        The measured value of the observable to compare to the emulator.
    :param combined_inv_cov:
        The inverse covariance matrix. Explicitly, the inverse of the sum of the mesurement covaraince matrix
        and the matrix from the emulator. Both are independent of emulator parameters, so can be precomputed.
    :return:
        The log liklihood of theta given the measurements and the emulator.
    """
    return 0
    '''
    # NOTE this could be generalized to emulate beyond wrt_r if I wanted
    y_bar = emu.emulate_wrt_r(dict(izip(param_names, theta)), r_bin_centers)[0]
    # should chi2 be calculated in log or linear?
    # answer: the user is responsible for taking the log before it comes here.
    delta = y_bar - y
    out = -0.5 * np.dot(delta, np.dot(combined_inv_cov, delta))
    return out
    '''

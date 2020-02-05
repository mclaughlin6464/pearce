import numpy as np
from itertools import izip
from multiprocessing import Pool
from scipy.linalg import inv
import emcee as mc
from pearce.mocks.kittens import TrainingBox
from collections import OrderedDict
import h5py

def lnprior(theta, param_names, param_bounds, *args):
    """
    Prior for an MCMC. Default is to assume flat prior for all parameters defined by the boundaries the
    emulator is built from. Retuns negative infinity if outside bounds or NaN
    :param theta:
        The parameters proposed by the sampler.
    :param param_names
        The names identifying the values in theta, needed to extract their boundaries
    :param param_bounds
        Dictionary of the boundaires allowed by each parameter.
    :return:
        Either 0 or -np.inf, depending if the params are allowed or not.
    """
    for p, t in izip(param_names, theta):
        low, high = param_bounds[p]

        if np.isnan(t) or t < low or t > high:
            return -np.inf
    return 0

def lnlike(theta, param_names, param_bounds, fixed_params, cat, r_bins, y, combined_inv_cov):
    """
    :param theta:
        Proposed parameters.
    :param param_names:
        The names of the parameters in theta
    :param fixed_params:
        Dictionary of parameters necessary to predict y_bar but are not being sampled over.
    :param cat:
        Cat object corresponding to the loaded cosmology.
    :param r_bins:
        The centers of the r bins y is measured in, angular or radial.
    :param y:
        The measured values of the observables to compare to the emulators. Must be an interable that contains
        predictions of each observable.
    :param combined_inv_cov:
        The inverse covariance matrices. Explicitly, the inverse of the sum of the mesurement covaraince matrix
        and the matrix from the emulator, both for each observable. Both are independent of emulator parameters,
         so can be precomputed. Must be an iterable with a matrixfor each observable.
    :return:
        The log liklihood of theta given the measurements and the emulator.
    """
    param_dict = dict(izip(param_names, theta))
    param_dict.update(fixed_params)

    cat.populate(param_dict)
    pred = cat.calc_vdf(r_bins, n_cores=1).squeeze()

    delta = pred - y
    #print delta
    return - np.dot(delta, np.dot(combined_inv_cov, delta))


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

def _random_initial_guess(param_bounds, nwalkers):
    """
    Create a random initial guess for the sampler. Creates a 3-sigma gaussian ball around the center of the prior space.
    :param param_names:
        The names of the parameters in the emulator
    :param nwalkers:
        Number of walkers to initiate. Must be the same as in resume_from_previous
    :param num_params:
        Number of params to initiate, must be the same as in resume_from_previous
    :return: pos0, the initial position of each walker for the chain.
    """

    pos0 = np.zeros((nwalkers, len(param_bounds)))
    for idx, (pname, pbound) in enumerate(param_bounds.iteritems()):
        low, high = pbound
        pos0[:, idx] = np.random.randn(nwalkers) * (np.abs(high - low) / 6.0) + (low + high) / 2.0

    return pos0

def run_mcmc_iterator(cat, param_bounds, y, cov, r_bins,fixed_params={},
                      pos0=None, nwalkers=100, nsteps=5000, ncores=8, return_lnprob=False):
    """
    Run an MCMC using emcee and the emu. Includes some sanity checks and does some precomputation.
    Also optimized to be more efficient than using emcee naively with the emulator.

    This version, as opposed to run_mcmc, "yields" each step of the chain, to write to file or to print.

    :param cat:
        Loaded version of a cat object with a model and halocatalog loaded
    :param param_names:
        Names of the parameters to constrain
    :param y:
        data to constrain against. either one array of observables, of size (n_bins*n_obs)
    :param cov:
        measured covariance of y for each y. Should have the same shape as y, but square
    :param r_bin_centers:
        The scale bins corresponding to all y in ys
    :param resume_from_previous:
        String listing filename of a previous chain to resume from. Default is None, which starts a new chain.
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
    :param return_lnprob:
        Whether to return the evaluation of lnprob on the samples along with the samples. Default is Fasle,
        which only returns samples.
    :yield:
        chain, collaposed to the shape ((nsteps-nburn)*nwalkers, len(param_names))
    """

    pool = Pool(processes=ncores)

    param_names = param_bounds.keys()
    num_params = len(param_names)
    combined_inv_cov = inv(cov)

    sampler = mc.EnsembleSampler(nwalkers, num_params, lnprob, pool=pool,
                                 args=(param_names, param_bounds, fixed_params, cat, r_bins, y, combined_inv_cov))


    if pos0 is None:
        pos0 = _random_initial_guess(param_names, nwalkers, num_params)

    for result in sampler.sample(pos0, iterations=nsteps, storechain=False):
        if return_lnprob:
            yield result[0], result[1]
        else:
            yield result[0]

if __name__ == "__main__":
    from sys import argv
    output_fname = argv[1]

    boxno = 12
    cat = TrainingBox(boxno)
    cat.load(1.0, HOD='zheng07')

    hod_param_bounds = OrderedDict({'logMmin': (13.0, 14.0),
                        'sigma_logM': (0.05, 0.5),
                        'alpha': (0.85, 1.15),
                        'logM0': (12.5, 14.5),
                        'logM1': (13.5, 15.5)} )

    true_point = _random_initial_guess(hod_param_bounds, 1)
    true_dict = OrderedDict(dict(zip(hod_param_bounds.keys(), true_point)))
    print 'Truth', true_dict
    cat.populate(true_dict)
    r_bins = np.logspace(-1, 1.6, 19)
    y = cat.calc_vdf(r_bins, ncores=1).squeeze()

    cov_ys = np.zeros((25, y.shape[0]))
    for i in xrange(25):
        cat.populate(true_dict)

        cov_ys[i] = cat.calc_vdf(r_bins, ncores=1).squeeze()

    with h5py.File(output_fname, 'w') as f:
        f.create_dataset('chain', (0, len(hod_param_bounds)),
                         compression = 'gzip', maxshape = (None, len(hod_param_bounds)))

        f.attrs['boxno'] = boxno
        f.attrs['r_bins'] = r_bins
        f.attrs['true_point'] = true_point
        f.attrs['hod_pnames'] = hod_param_bounds.keys()
        f.attrs['cov'] = covmat
        f.attrs['y'] = y

    covmat = np.cov(cov_ys, rowvar=False)
    nwalkers = 100
    nsteps = 5000

    for step, pos in enumerate(run_mcmc_iterator(cat, hod_param_bounds, y, covmat, r_bins,\
                                                 nwalkers=nwalkers, nsteps=nsteps, ncores=8, return_lnprob=False)):

        with h5py.File(output_fname, 'a') as f:

            chain_dset, like_dset = f['chain'], f['lnprob']
            l = len(chain_dset)
            chain_dset.resize((l + nwalkers), axis=0)

            chain_dset[-nwalkers:] = pos[0]


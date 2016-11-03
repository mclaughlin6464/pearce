#!/bin/bash
'''The Emu object esentially wraps the George gaussian process code. It handles building, training, and predicting.'''

from os import path
from time import time
import warnings
from glob import glob
from itertools import izip
from multiprocessing import cpu_count

import numpy as np
import scipy.optimize as op
from scipy.linalg import inv
from scipy.interpolate import interp1d
import george
from george.kernels import *
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
import emcee as mc

from .trainingData import GLOBAL_FILENAME
from .ioHelpers import global_file_reader, obs_file_reader, parameter

#TODO abstract base classes? Depends on what methods need to be subclassed.
#TODO methods I think need to be reordered

class Emu(object):
    #TODO docstrings? here and init.
    valid_methods = {'gp', 'svr', 'gbdt', 'rf', 'krr'}  # could add more, coud even check if they exist in sklearn

    def __init__(self, training_dir,method = 'gp',hyperparams={}, params=None, fixed_params={}, independent_variable=None):

        assert method in self.valid_methods

        if independent_variable == 'bias':
            raise NotImplementedError("I have to work on how to do xi_mm first.")
        assert independent_variable in {None, 'r2'}  # no bias for now.

        # TODO I hate the assembly bias parameter keys. It'd be nice if the use could pass in something
        # else and I make a change
        # use default if needed
        if params is None:
            from .ioHelpers import DEFAULT_PARAMS as params

        self.method = method
        #TODO store hyperparams?
        self.ordered_params = params
        self.fixed_params = fixed_params
        self.independent_variable = independent_variable

        self.get_training_data(training_dir)
        self.build_emulator(hyperparams)

    # I can get LHC from the training data. If any coordinate equals any other in its column we know!
    def get_training_data(self, training_dir):
        '''
        Read the training data for the emulator and attach it to the object.
        :param training_dir:
            Directory where training data from trainginData is stored.
        :param fixed_params:
            Parameters to hold fixed. Only available if data in training_dir is a full hypercube, not a latin hypercube.
        :return: None
        '''
        x, y, yerr = self.load_data(training_dir, self.fixed_params, self.independent_variable)

        self.x = x
        self.y = y
        self.yerr = yerr
        self.y_hat = np.zeros(self.y.shape[1]) if len(y.shape)>1 else 0# self.y.mean(axis = 0)
        self.y -= self.y_hat

        ndim = self.x.shape[1]
        self.fixed_ndim = len(self.fixed_params)
        self.emulator_ndim = ndim  # The number of params for the emulator is different than those in sampling.
        self.sampling_ndim = ndim - 1

    #@abstractMethod
    def load_data(self, data_dir, fixed_params, independent_variable):
        '''Implemented in subclasses'''
        raise NotImplementedError

    def _iv_transform(self, independent_variable, obs, cov):
        '''
        Independent variable tranform. Helper function that consolidates this operation all in one place.
        :param independent_variable:
            Which iv to transform to. Current optins are None (just take log) and r2.
        :param obs:
            Observable to transform (xi, wprp, etc.)
        :param cov:
            Covariance of obs
        :return:
            y, yerr the transformed iv's for the emulator
        '''
        if independent_variable is None:
            y = np.log10(obs)
            # Approximately true, may need to revisit
            # yerr[idx * NBINS:(idx + 1) * NBINS] = np.sqrt(np.diag(cov)) / (xi * np.log(10))
            y_err = np.sqrt(np.diag(cov)) / (
                obs * np.log(10))  # I think this is right, extrapolating from the above.
        elif independent_variable == 'r2':  # r2
            y = obs * self.bin_centers * self.bin_centers
            y_err = np.sqrt(np.diag(cov)) * self.bin_centers  # I think this is right, extrapolating from the above.
        else:
            raise ValueError('Invalid independent variable %s' % independent_variable)

        '''
        if independent_variable == 'bias':
            y[idx * NBINS:(idx + 1) * NBINS] = xi / xi_mm
            ycovs.append(cov / np.outer(xi_mm, xi_mm))
        '''

        return y, y_err

    def _sort_params(self, t, argsort=False):
        '''
        Sort the parameters in a defined away given the orderering.
        :param t:
            Parameter vector to sort. Should have dims (N, N_params) and be in the order
            defined by ordered_params
        :param argsort:
            If true, return indicies that would sort the array rather than the sorted array itself.
            Default is False.
        :return:
            If not argsort, returns the sorted array by column and row. 
            If argsort, return the indicies that would sort the array.
        '''
        if t.shape[0] == 1:
            if argsort:
                return np.array([0])
            return t  # a row array is already sorted!

        if argsort:  # returns indicies that would sort the array
            # weird try structure because this view is very tempermental!
            try:
                idxs = np.argsort(t.view(','.join(['float64' for _ in xrange(min(t.shape))])),
                                  order=['f%d' % i for i in xrange(min(t.shape))], axis=0)
            except ValueError:  # sort with other side
                idxs = np.argsort(t.view(','.join(['float64' for _ in xrange(max(t.shape))])),
                                  order=['f%d' % i for i in xrange(max(t.shape))], axis=0)

            return idxs[:, 0]

        try:
            t = np.sort(t.view(','.join(['float64' for _ in xrange(min(t.shape))])),
                        order=['f%d' % i for i in xrange(min(t.shape))], axis=0).view(np.float)
        except ValueError:  # sort with other side
            t = np.sort(t.view(','.join(['float64' for _ in xrange(max(t.shape))])),
                        order=['f%d' % i for i in xrange(max(t.shape))], axis=0).view(np.float)

        return t

    def build_emulator(self,hyperparams):
        '''
        Initialization of the emulator from recovered training data. Calls submethods depending on "method"
        :param method:
            The machine learning method to use.
.       :param hyperparams
            A dictionary of hyperparameter kwargs for the emulator
        :param fixed_params:
            Parameterst to hold fixed in teh training data
        :return: None
        '''

        if self.method == 'gp':
            self._build_gp(hyperparams)
        else: #an sklearn method
            self._build_skl(hyperparams)

    # @abstractMethod
    def _build_gp(self, hyperparams):
        """
        Implemented in subclasses
        """
        raise NotImplementedError

    # @abstractMethod
    def _build_skl(self, hyperparams):
        """
        Implemented in subclasses
        """
        raise NotImplementedError

    #TODO undserscore?
    def get_initial_guess(self, independent_variable):
        '''
        Return the initial guess for the emulator, based on what the iv is. Guesses are learned from
        previous experiments.
        :param independent_variable:
            Which variable to return the guesses for.
        :param fixed_params:
            Parameters to hold fixed; only return guess for parameters that are not fixed.
        :return: initial_guesses, a dictionary of the guess for each parameter
        '''

        # default
        ig = {'amp': 1}
        ig.update({p.name: 0.1 for p in self.ordered_params})

        if self.obs == 'xi':
            if independent_variable is None:
                ig = {'amp': 0.481, 'logMmin': 0.1349, 'sigma_logM': 0.089,
                      'logM0': 2.0, 'logM1': 0.204, 'alpha': 0.039,
                      'f_c': 0.041, 'r': 0.040}
            else:
                pass
        elif self.obs == 'wp':
            if independent_variable is None:
                ig = {'logMmin': 1.7348042925, 'f_c': 0.327508062386, 'logM0': 15.8416094906,
                      'sigma_logM': 5.36288382789, 'alpha': 3.63498762588, 'r': 0.306139450843,
                      'logM1': 1.66509412286, 'amp': 1.18212664544}
        else:
            pass  # no other guesses saved yet.

        # remove entries for variables that are being held fixed.
        for key in fixed_params.iterkeys():
            del ig[key]

        return ig

    def _make_kernel(self,metric):
        """
        Helper method to build a george kernel for GP's and kernel-based regressions.
        :param metric:
            Hyperparams for kernel determining relative length scales and amplitudes
        :return:
            A george ExpSquredKernel object with this metric
        """

        if not metric:
            ig = self.get_initial_guess(self.independent_variable, self.fixed_params)
        else:
            ig = metric  # use the user's initial guesses

        #TODO does this need to be attached? is it ever used again
        metric = [ig['amp']]
        for p in self.ordered_params:
            if p.name in self.fixed_params:
                continue
            try:
                self.metric.append(ig[p.name])
            except KeyError:
                raise KeyError('Key %s was not in the metric.' % p.name)

        metric = np.array(metric)

        a = metric[0]
        #TODO other kernels?
        return a * ExpSquaredKernel(metric[1:], ndim=self.emulator_ndim)

    # Not sure this will work at all in an LHC scheme.
    #TODO can i use load_data here?
    def get_plot_data(self, em_params, training_dir, independent_variable=None, fixed_params={}):
        '''
        Similar function to get_training_data. However, returns values for plotting comparisons to the emulator.
        :param em_params:
            Similar to fixed params. A dictionary of values held fixed in the emulator, as opposed to fixed_params
            which are values held fixed in the training data.
        :param training_dir:
            Directory where training data from trainginData is stored.
        :param independent_variable:
            Independant variable to emulate. Options are xi, r2xi, and bias (eventually).
        :param fixed_params:
            Parameters to hold fixed. Only available if data in training_dir is a full hypercube, not a latin hypercube.
        :return: log_r, y, yerr for the independent variable at the points specified by fixed nad em params.
        '''
        assert len(em_params) + len(fixed_params) - len(self.ordered_params) <= 1  # can exclude r

        bins, cosmo_params, obs, method = global_file_reader(path.join(training_dir, GLOBAL_FILENAME))

        if method == 'LHC':
            raise ValueError('The data in training_dir is form a Latin Hypercube. \
                                        It is not possible to get plot data from this data. ')

        obs_files = sorted(glob(path.join(training_dir, 'obs*.npy')))
        cov_files = sorted(
            glob(path.join(training_dir, 'cov*.npy')))  # since they're sorted, they'll be paired up by params.
        bin_centers = (bins[:-1] + bins[1:]) / 2
        nbins = bin_centers.shape[0]

        npoints = len(obs_files)  # each file contains NBINS points in r, and each file is a 6-d point

        log_bin_centers = np.zeros((npoints, nbins))
        y = np.zeros((npoints, nbins))
        yerr = np.zeros((npoints, nbins))

        x = np.zeros((npoints, len(em_params)))

        warned = False
        num_skipped = 0
        num_used = 0
        for idx, (obs_file, cov_file) in enumerate(izip(obs_files, cov_files)):
            params, obs, cov = obs_file_reader(obs_file, cov_file)

            # skip values that aren't where we've fixed them to be.
            # It'd be nice to do this before the file I/O. Not possible without putting all info in the filename.
            # or, a more nuanced file structure
            # TODO check if a fixed_param is not one of the options
            if any(params[key] != val for key, val in fixed_params.iteritems()):
                continue

            # same as the above for emulation params
            # more complex since em_params can have a float or array
            to_continue = True
            for key, val in em_params.iteritems():
                if type(val) is type(y):
                    if np.all(np.abs(params[key] - val) > 1e-3):
                        break
                elif np.abs(params[key] - val) > 1e-3:
                    break
            else:
                to_continue = False  # no break

            if to_continue:
                continue

            if np.any(np.isnan(cov)) or np.any(np.isnan(obs)):
                if not warned:
                    warnings.warn('WARNING: NaN detected. Skipping point in %s' % cov_file)
                    warned = True
                num_skipped += 1
                continue

            num_used += 1

            log_bin_centers[idx] = np.log10(bin_centers)
            y[idx], yerr[idx] = self._iv_transform(independent_variable, obs, cov)

            x[idx] = np.array([params[p.name] for p in self.ordered_params if p.name in params])

        # remove rows that were skipped due to the fixed thing
        # NOTE: HACK
        # a reshape may be faster.
        zeros_slice = np.all(y != 0.0, axis=1)

        print x[zeros_slice].shape

        sort_idxs = self._sort_params(x[zeros_slice], argsort=True)

        return log_bin_centers[zeros_slice][sort_idxs], y[zeros_slice][sort_idxs], yerr[zeros_slice][sort_idxs]

    def train_metric(self, **kwargs):
        raise NotImplementedError

    def emulate(self, em_params, gp_errs=False):
        raise NotImplementedError

    def emulate_wrt_r(self, em_params, bin_centers, gp_err=False):
        raise NotImplementedError

    #TODO this feature is not super useful anymore, and also is poorly defined w.r.t non gp methods.
    #did a lot of work on it tho, maybe i'll leave it around...?
    #TODO change name to LOO
    #TODO will break with the way I'm changing EC.
    def _jackknife_errors(self, y, t):
        '''
        Calculate the LOO Jackknife error matrix. This is implemented using the analytic LOO procedure,
        which is much faster than re-doing an inversion for each sample. May be useful if the GP's matrix is not
        accurate.
        :param y:
            Values of the independent variable for the training points, used in the prediction.
        :param t:
            Values of the dependant variables to predict at.
        :return:
            jk_cov: a covariance matrix with the dimensions of cov.
        '''
        # from time import time

        assert self.method == 'gp'

        if isinstance(self, ExtraCrispy):
            emulator = self.emulators[0] #hack for EC, do somethign smarter later
        else:
            emulator = self.emulator

        # We need to perform one full inverse to start.
        K_inv_full = emulator.solver.apply_inverse(np.eye(emulator._alpha.size),
                                                  in_place=True)

        # TODO deepcopy?
        x = self.x[:]

        N = K_inv_full.shape[0]

        mus = np.zeros((N, t.shape[0]))
        # t0 = time()

        # iterate over training points to leave out
        for idx in xrange(N):
            # swap the values of the LOO point and the last point.
            x[[N - 1, idx]] = x[[idx, N - 1]]
            y[[N - 1, idx]] = y[[idx, N - 1]]

            K_inv_full[[idx, N - 1], :] = K_inv_full[[N - 1, idx], :]
            K_inv_full[:, [idx, N - 1]] = K_inv_full[:, [N - 1, idx]]

            # the inverse of the LOO GP
            # formula found via MATH
            K_m_idx_inv = K_inv_full[:N - 1, :][:, :N - 1] \
                          - np.outer(K_inv_full[N - 1, :N - 1], K_inv_full[:N - 1, N - 1]) / K_inv_full[N - 1, N - 1]

            alpha_m_idx = np.dot(K_m_idx_inv, y[:N - 1] - emulator.mean(x[:N - 1]))

            Kxxs_t = emulator.kernel.value(t, x[:N - 1])

            # Store the estimate for this LOO GP
            mus[idx, :] = np.dot(Kxxs_t, alpha_m_idx) + emulator.mean(t)

            # print mus[idx]
            # print

            # restore the original values for the next loop
            x[[N - 1, idx]] = x[[idx, N - 1]]
            y[[N - 1, idx]] = y[[idx, N - 1]]

            K_inv_full[[idx, N - 1], :] = K_inv_full[[N - 1, idx], :]
            K_inv_full[:, [idx, N - 1]] = K_inv_full[:, [N - 1, idx]]

        # print time() - t0, 's Total'
        # return the jackknife cov matrix.
        cov = (N - 1.0) / N * np.cov(mus, rowvar=False)
        if mus.shape[1] == 1:
            return np.array([[cov]])  # returns float in this case
        else:
            return cov

    def goodness_of_fit(self, truth_dir, N=None, statistic='r2'):
        '''
        Calculate the goodness of fit of an emulator as compared to some validation data.
        :param truth_dir:
            Directory structured similary to the training data, but NOT used for training.
        :param N:
            Number of points to use to calculate G.O.F. measures. "None" tests against all values in truth_dir. If N
            is less than the number of points, N are randomly selected.
        :param statistic:
            What G.O.F. statistic to calculate. Default is R2. Other options are rmsfd, abs(olute), and rel(ative).
        :return: values, a numpy arrray of the calculated statistics at each of the N training opints.
        '''
        assert statistic in {'r2', 'rmsfd', 'abs', 'rel'}
        assert N > 0 and int(N) == N

        x, y = self.load_data(truth_dir, self.fixed_params, self.independent_variable)

        np.random.seed(int(time()))

        if N is not None: #make a random choice
            idxs = np.random.choice(x.shape[0], N, replace=False)

            x, y = x[idxs], y[idxs]

        log_y = np.log10(y)

        bins, _, _, _ = global_file_reader(path.join(truth_dir, GLOBAL_FILENAME))
        bin_centers = (bins[1:] + bins[:-1]) / 2
        params = {p.name: x[:, i] for p, i in zip(self.ordered_params, xrange(x.shape[1])) if
                  p.name not in self.fixed_params}

        #probably should consider not having wrt r here.
        pred_log_y = self.emulate_wrt_r(params, bin_centers)

        if statistic == 'rmsfd':
            return np.sqrt(np.mean(( ((pred_log_y - log_y) ** 2) / (log_y ** 2)), axis=0) )

        elif statistic == 'rms':
            return np.sqrt(np.mean(( (pred_log_y - log_y) ** 2), axis=0) )

        #TODO sklearn methods can do this themselves. But i've already tone the prediction!
        elif statistic == 'r2':  # r2
            SSR = np.sum((pred_log_y - log_y) ** 2, axis=0)
            SST = np.sum((log_y - log_y.mean(axis=0)) ** 2, axis = 0)

            return 1 - SSR / SST

        elif statistic == 'abs':
            return np.mean( (pred_log_y - log_y), axis = 0)
        else:  # 'rel'
            return np.mean( (pred_log_y- log_y ) / log_y, axis = 0)

    def estimate_uncertainty(self, truth_dir, N=None):
        '''
        Estimate the uncertainty of the emulator by comparing to a "test" box of true values.
        :param truth_dir:
            Name of a directory of true test values, of the same format as the train_dir
        :param N:
            Number of points to compare to. If None (default) will use all points. Else will select random sample.
        :return:
            covariance matrix with dim n_binsxn_bins. Will only have diagonal elemtns of est. uncertainties.
        '''
        rms_err = self.goodness_of_fit(truth_dir, N, statistic='rms')

        return np.diag(rms_err)

    def run_mcmc(self, y, cov, bin_centers, nwalkers=1000, nsteps=100, nburn=20, n_cores='all'):
        '''
        Run an MCMC sampler, using the emulator. Uses emcee to perform sampling.
        :param y:
            A true y value to recover the parameters of theta. NOTE: The emulator emulates some indepedant variables in 
            log space, others in linear. Make sure y is in the same space!
        :param cov:
            The measurement covariance matrix of y
        :param bin_centers:
            The centers of the bins y is measured in (radial or angular).
        :param nwalkers:
            Optional. Number of walkers for emcee. Default is 1000.
        :param nsteps:
            Optional. Number of steps for emcee. Default is 100.
        :param nburn:
            Optional. Number of burn-in steps for emcee. Default is 20.
        :param n_cores:
            Number of cores, either an iteger or 'all'. Default is 'all'.
        :return:
            chain, a numpy array of the sample chain.
        '''

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
        assert y.shape[0] == bin_centers.shape[0]

        sampler = mc.EnsembleSampler(nwalkers, self.sampling_ndim, lnprob,
                                     threads=n_cores, args=(self, y, cov, bin_centers))

        pos0 = np.zeros((nwalkers, self.sampling_ndim))
        # The zip ensures we don't use the params that are only for the emulator
        for idx, (p, _) in enumerate(izip(self.ordered_params, xrange(self.sampling_ndim))):
            # pos0[:, idx] = np.random.uniform(p.low, p.high, size=nwalkers)
            pos0[:, idx] = np.random.normal(loc=(p.high + p.low) / 2, scale=(p.high + p.low) / 10, size=nwalkers)

        sampler.run_mcmc(pos0, nsteps)

        # Note, still an issue of param label ordering here.
        chain = sampler.chain[:, nburn:, :].reshape((-1, self.sampling_ndim))

        return chain


# These functions cannot be instance methods
# Emcee throws a few when trying to compile the liklihood functions that are attached
# to the object calling it
def lnprob(theta, *args):
    '''
    The total liklihood for an MCMC. Sadly, can't be an instance of the Emu Object.
    :param theta:
        Parameters for the proposal
    :param args:
        Arguments to pass into the liklihood
    :return:
        Log Liklihood of theta, a float.
    '''
    lp = lnprior(theta, *args)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, *args)


def lnprior(theta, emu, *args):
    '''
    Prior for an MCMC. Currently asserts theta is between the boundaries used to make the emulator.
    Could do something more clever later.
    :param theta:
        The parameters proposed by the sampler.
    :param emu:
        The emulator object. Needs to be accessed to get the priors.
    :return:
        Either 0 or -np.inf, depending if the params are allowed or not.
    '''
    return 0 if all(p.low < t < p.high for p, t in izip(emu.ordered_params, theta)) else -np.inf


def lnlike(theta, emu, y, cov, bin_centers):
    '''
    The liklihood of parameters theta given the other parameters and the emulator.
    :param theta:
        Proposed parameters.
    :param emu:
        The emulator object. Used to perform the emulation.
    :param y:
        The measured value of the observable to compare to the emulator.
    :param cov:
        The covariance matrix of the measured values.
    :param bin_centers:
        The centers of the bins y is measured in, angular or radial.
    :return:
        The log liklihood of theta given the measurements and the emulator.
    '''
    em_params = {p.name: t for p, t in zip(emu.ordered_params, theta)}

    # using my own notation
    y_bar, G = emu.emulate_wrt_r(em_params, bin_centers)
    # should chi2 be calculated in log or linear?
    # answer: the user is responsible for taking the log before it comes here.

    D = G + cov
    delta = y_bar - y
    chi2 = -0.5 * np.dot(delta, np.dot(inv(D), delta))
    return chi2


class OriginalRecipe(Emu):
    '''Emulator that emulates with bins as an implicit parameter. '''

    def load_data(self, data_dir, fixed_params, independent_variable):
        '''
        Read data in the format compatible with this object and return it
        :param data_dir:
            Directory where data from trainingData is stored
        :param fixed_params:
            Parameters to hold fixed. Only available if data in data_dir is a full hypercube, not a latin hypercube
        :param independent_variable:
            Independant variable to emulate. Options are xi, r2xi, and bias (eventually)..
        :return: None
        '''

        bins, cosmo_params, obs, method = global_file_reader(path.join(data_dir, GLOBAL_FILENAME))

        if fixed_params and method == 'LHC':
            raise ValueError('Fixed parameters is not empty, but the data in data_dir is form a Latin Hypercube. \
                                Cannot performs slices on a LHC.')

        self.obs = obs

        obs_files = sorted(glob(path.join(data_dir, 'obs*.npy')))
        cov_files = sorted(
            glob(path.join(data_dir, 'cov*.npy')))  # since they're sorted, they'll be paired up by params.
        self.bin_centers = (bins[:-1] + bins[1:]) / 2
        nbins = self.bin_centers.shape[0]

        npoints = len(obs_files) * nbins  # each file contains NBINS points in r, and each file is a 6-d point

        varied_params = set([p.name for p in self.ordered_params]) - set(fixed_params.keys())
        ndim = len(varied_params)  # lest we forget r

        x = np.zeros((npoints, ndim))
        y = np.zeros((npoints,))
        yerr = np.zeros((npoints,))

        warned = False
        num_skipped = 0
        num_used = 0
        for idx, (obs_file, cov_file) in enumerate(izip(obs_files, cov_files)):
            params, obs, cov = obs_file_reader(obs_file, cov_file)

            # skip values that aren't where we've fixed them to be.
            # It'd be nice to do this before the file I/O. Not possible without putting all info in the filename.
            # or, a more nuanced file structure
            # TODO check if a fixed_param is not one of the options i.e. typo
            if any(params[key] != val for key, val in fixed_params.iteritems()):
                continue

            if np.any(np.isnan(cov)) or np.any(np.isnan(obs)):
                if not warned:
                    warnings.warn('WARNING: NaN detected. Skipping point in %s' % cov_file)
                    warned = True
                num_skipped += 1
                continue

            num_used += 1

            # doing some shuffling and stacking
            file_params = []
            # NOTE could do a param ordering here
            for p in self.ordered_params:
                if p.name in fixed_params:
                    continue
                # TODO change 'r' to something else.
                if p.name == 'r':
                    file_params.append(np.log10(self.bin_centers))
                else:
                    file_params.append(np.ones((nbins,)) * params[p.name])

            x[idx * nbins:(idx + 1) * nbins, :] = np.stack(file_params).T
            # TODO the time has come to do something smarter for bias... I will ignore for now.

            y[idx * nbins:(idx + 1) * nbins], yerr[idx * nbins:(idx + 1) * nbins] = self._iv_transform(
                independent_variable, obs, cov)

        # ycov = block_diag(*ycovs)
        # ycov = np.sqrt(np.diag(ycov))

        # remove rows that were skipped due to the fixed thing
        # NOTE: HACK
        # a reshape may be faster.
        zeros_slice = np.all(x != 0.0, axis=1)
        # set the results of these calculations.

        return x[zeros_slice], y[zeros_slice], yerr[zeros_slice]

    def _build_gp(self, hyperparams):
        '''
        Initialize the GP emulator.
        :param hyperparams:
            Key word parameters for the emulator
        :return: None
        '''
        #TODO could use more of the hyperparams...
        metric = hyperparams['metric'] if 'metric' in hyperparams else {}
        kernel = self._make_kernel(metric)
        # TODO is it confusing for this to have the same name as the sklearn object with a different API?
        # maybe it should be a property? or private?
        self.emulator = george.GP(kernel)
        # gp = george.GP(kernel, solver=george.HODLRSolver, nleaf=x.shape[0]+1,tol=1e-18)

        self.emulator.compute(self.x, self.yerr, sort=False)  # NOTE I'm using a modified version of george!

    def _build_skl(self, hyperparams):
        """
        Build a scikit learn emulator
        :param hyperparams:
            Key word parameters for the emulator
        :return: None
        """
        skl_methods = {'gbdt': GradientBoostingRegressor, 'rf': RandomForestRegressor, \
                       'svr': SVR, 'krr': KernelRidge}

        if self.method in {'svr', 'krr'}:  # kernel based method
            metric = hyperparams['metric'] if 'metric' in hyperparams else {}
            kernel = self._make_kernel(metric)
            if 'metric' in hyperparams:
                del hyperparams['metric']
            if self.method == 'svr':  # slight difference in these, sadly
                hyperparams['kernel'] = kernel.value
            else:  # krr
                hyperparams['kernel'] = lambda x1, x2: kernel.value(np.array([x1]), np.array([x2]))

        self.emulator = skl_methods[self.method](**hyperparams)
        self.emulator.fit(self.x, self.y)

    def train_metric(self, **kwargs):
        '''
        Train the metric parameters of the GP. Has a spotty record of working.
        Best used as used in lowDimTraining.
        If attempted to be used with an emulator that is not GP, will raise an error.
        :param kwargs:
            Kwargs that will be passed into the scipy.optimize.minimize
        :return: success: True if the training was successful.
        '''

        #TODO kernel based methods may want to use this...
        assert self.method == 'gp'

        # move these outside? hm.
        def nll(p):
            # Update the kernel parameters and compute the likelihood.
            # params are log(a) and log(m)
            self.emulator.kernel[:] = p
            ll = self.emulator.lnlikelihood(self.y, quiet=True)

            # The scipy optimizer doesn't play well with infinities.
            return -ll if np.isfinite(ll) else 1e25

        # And the gradient of the objective function.
        def grad_nll(p):
            # Update the kernel parameters and compute the likelihood.
            self.emulator.kernel[:] = p
            return -self.emulator.grad_lnlikelihood(self.y, quiet=True)

        p0 = self.emulator.kernel.vector
        results = op.minimize(nll, p0, jac=grad_nll, **kwargs)
        # results = op.minimize(nll, p0, jac=grad_nll, method='TNC', bounds =\
        #   [(np.log(0.01), np.log(10)) for i in xrange(ndim+1)],options={'maxiter':50})
        print results

        self.emulator.kernel[:] = results.x
        self.emulator.recompute()
        #self.metric = np.exp(results.x)

        return results.success

    def emulate(self, em_params, gp_errs = False):
        '''
        Perform predictions with the emulator.
        :param em_params:
            Dictionary of what values to predict at for each param. Values can be
            an array or a float.
        :param gp_errs:
            Boolean, decide whether or not to return the errors from the gp prediction. Default is False.
            Will throw error if method is not gp.
        :return: mu, cov. The predicted value and the covariance matrix for the predictions
        '''

        if gp_errs:
            assert self.method == 'gp' #only has meaning for gp's

        input_params = {}
        input_params.update(self.fixed_params)
        input_params.update(em_params)
        assert len(input_params) == self.emulator_ndim + self.fixed_ndim  # check dimenstionality
        for i in input_params:  # check that the names in input params are all defined in the ordering.
            assert any(i == p.name for p in self.ordered_params)

        # i'd like to remove 'r'. possibly requiring a passed in param?
        t_list = [input_params[p.name] for p in self.ordered_params if p.name in em_params]
        t_grid = np.meshgrid(*t_list)
        t = np.stack(t_grid).T
        # TODO george can sort?
        t = t.reshape((-1, self.emulator_ndim))

        t = self._sort_params(t)

        if self.method == 'gp':
            return self.emulator.predict(self.y, t, mean_only = not gp_errs)
        else:
            return self.emulator.predict(t)

    # TODO It's not clear to the user if bin_centers should be log or not!
    def emulate_wrt_r(self, em_params, bin_centers, gp_errs=False):
        '''
        Conveniance function. Add's 'r' to the emulation automatically, as this is the
        most common use case.
        :param em_params:
            Dictionary of what values to predict at for each param. Values can be array
            or float.
        :param bin_centers:
            Centers of bins to predict at, for each point in HOD-space.
        :return:
        '''
        vep = dict(em_params)
        # TODO change 'r' to something more general
        vep.update({'r': np.log10(bin_centers)})
        out = self.emulate(vep, gp_errs)
        return out


class ExtraCrispy(Emu):
    '''Emulator that emulates with bins as an implicit parameter. '''

    def load_data(self, data_dir, fixed_params, independent_variable):
        '''
        Read the training data for the emulator and attach it to the object.
        :param data_dir:
            Directory where training data from trainginData is stored.
        :param independent_variable:
            Independant variable to emulate. Options are xi, r2xi, and bias (eventually).
        :param fixed_params:
            Parameters to hold fixed. Only available if data in data_dir is a full hypercube, not a latin hypercube.
        :return: None
        '''

        bins, cosmo_params, obs, method = global_file_reader(path.join(data_dir, GLOBAL_FILENAME))

        if fixed_params and method == 'LHC':
            raise ValueError('Fixed parameters is not empty, but the data in data_dir is form a Latin Hypercube. \
                                Cannot performs slices on a LHC.')

        self.obs = obs

        obs_files = sorted(glob(path.join(data_dir, 'obs*.npy')))
        cov_files = sorted(
            glob(path.join(data_dir, 'cov*.npy')))  # since they're sorted, they'll be paired up by params.
        self.bin_centers = (bins[:-1] + bins[1:]) / 2
        nbins = self.bin_centers.shape[0]
        npoints = len(obs_files)  # each file contains NBINS points in r, and each file is a 6-d point

        varied_params = set(self.ordered_params) - set(fixed_params.keys())
        ndim = len(varied_params)  # lest we forget r

        # not sure about this.

        x = np.zeros((npoints, ndim))
        y = np.zeros((npoints, nbins))
        yerr = np.zeros((npoints, nbins))

        warned = False
        num_skipped = 0
        num_used = 0
        for idx, (obs_file, cov_file) in enumerate(izip(obs_files, cov_files)):

            params, obs, cov = obs_file_reader(obs_file, cov_file)

            # skip values that aren't where we've fixed them to be.
            # It'd be nice to do this before the file I/O. Not possible without putting all info in the filename.
            # or, a more nuanced file structure
            # TODO check if a fixed_param is not one of the options i.e. typo
            if any(params[key] != val for key, val in fixed_params.iteritems()):
                continue

            if np.any(np.isnan(cov)) or np.any(np.isnan(obs)):
                if not warned:
                    warnings.warn('WARNING: NaN detected. Skipping point in %s' % cov_file)
                    warned = True
                num_skipped += 1
                continue

            num_used += 1

            # doing some shuffling and stacking
            file_params = []
            # NOTE could do a param ordering here
            for p in self.ordered_params:
                if p.name in fixed_params:
                    continue
                file_params.append(params[p.name])

            x[idx, :] = np.stack(file_params).T

            y[idx], yerr[idx] = self._iv_transform(independent_variable, obs, cov)

        # ycov = block_diag(*ycovs)
        # ycov = np.sqrt(np.diag(ycov))

        # remove rows that were skipped due to the fixed thing
        # NOTE: HACK
        # a reshape may be faster.
        zeros_slice = np.all(x != 0.0, axis=1)

        return x[zeros_slice], y[zeros_slice, :], yerr[zeros_slice, :]

    def _build_gp(self, hyperparams):
        '''
        Initialize the GP emulator.
        :param hyperparams:
            Key word parameters for the emulator
        :return: None
        '''
        #TODO could use more of the hyperparams...
        metric = hyperparams['metric'] if 'metric' in hyperparams else {}
        kernel = self._make_kernel(metric)
        # TODO is it confusing for this to have the same name as the sklearn object with a different API?
        # maybe it should be a property? or private?
        emulator = george.GP(kernel)
        # gp = george.GP(kernel, solver=george.HODLRSolver, nleaf=x.shape[0]+1,tol=1e-18)

        emulator.compute(self.x, np.zeros_like(self.yerr), sort=False)  # NOTE I'm using a modified version of george!

        #For EC, i'm storing an emulator per bin.
        #I'll have to thikn about how to differ the hyperparams.
        #For now, it'll replicate the same behavior as before.
        #TODO not happy, in general, EC has "emulators" not "emulator" like the others.
        #Arguement would be this should be all abstracted out from the user.
        self.emulators = [emulator for i in xrange(self.yerr.shape[1])]

    def _build_skl(self, hyperparams):
        """
        Build a scikit learn emulator
        :param hyperparams:
            Key word parameters for the emulator
        :return: None
        """
        skl_methods = {'gbdt': GradientBoostingRegressor, 'rf': RandomForestRegressor, \
                       'svr': SVR, 'krr': KernelRidge}

        #Same kernel concerns as above.
        if self.method in {'svr', 'krr'}:  # kernel based method
            metric = hyperparams['metric'] if 'metric' in hyperparams else {}
            kernel = self._make_kernel(metric)
            if 'metric' in hyperparams:
                del hyperparams['metric']
            if self.method == 'svr':  # slight difference in these, sadly
                hyperparams['kernel'] = kernel.value
            else:  # krr
                hyperparams['kernel'] = lambda x1, x2: kernel.value(np.array([x1]), np.array([x2]))


        self.emulators = [skl_methods[self.method](**hyperparams) for i in xrange(self.yerr.shape[1]) ]
        for y, emulator in zip(self.y.T, self.emulators):
            emulator.fit(self.x, y)

    def train_metric(self, **kwargs):
        '''
        Train the emulator. Has a spotty record of working. Better luck may be had with the NAMEME code.
        :param kwargs:
            Kwargs that will be passed into the scipy.optimize.minimize
        :return: success: True if the training was successful.
        '''

        assert self.method == 'gp'

        #emulators is a list containing refernces to the same object. this should still work!
        emulator = self.emulators[0]

        # move these outside? hm.
        def nll(p):
            # Update the kernel parameters and compute the likelihood.
            # params are log(a) and log(m)
            emulator.kernel[:] = p
            # check this has the right direction
            ll = np.sum(emulator.lnlikelihood(y, quiet=True) for y in self.y)

            # The scipy optimizer doesn't play well with infinities.
            return -ll if np.isfinite(ll) else 1e25

        # And the gradient of the objective function.
        def grad_nll(p):
            # Update the kernel parameters and compute the likelihood.
            emulator.kernel[:] = p
            # mean or sum?
            return -np.mean(emulator.grad_lnlikelihood(y, quiet=True) for y in self.y)

        p0 = emulator.kernel.vector
        results = op.minimize(nll, p0, jac=grad_nll, **kwargs)
        # results = op.minimize(nll, p0, jac=grad_nll, method='TNC', bounds =\
        #   [(np.log(0.01), np.log(10)) for i in xrange(ndim+1)],options={'maxiter':50})

        emulator.kernel[:] = results.x
        emulator.recompute()

        return results.success

    def emulate(self, em_params, gp_errs=False):
        '''
        Perform predictions with the emulator.
        :param varied_em_params:
            Dictionary of what values to predict at for each param. Values can be
            an array or a float.
        :return: mu, cov. The predicted value and the covariance matrix for the predictions
        '''
        if gp_errs:
            assert self.method == 'gp'

        input_params = {}
        input_params.update(self.fixed_params)
        input_params.update(em_params)
        assert len(input_params) == self.emulator_ndim + self.fixed_ndim  # check dimenstionality
        for i in input_params:
            assert any(i == p.name for p in self.ordered_params)

        # i'd like to remove 'r'. possibly requiring a passed in param?
        t_list = [input_params[p.name] for p in self.ordered_params if p.name in em_params]
        t_grid = np.meshgrid(*t_list)
        t = np.stack(t_grid).T
        # should add a fixed_param dim
        t = t.reshape((-1, self.emulator_ndim))
        t = self._sort_params(t)

        all_mu = np.zeros((t.shape[0], self.y.shape[1]))  # t down nbins across
        # all_err = np.zeros((t.shape[0], self.y.shape[1]))
        all_cov = []  # np.zeros((t.shape[0], t.shape[0], self.y.shape[1]))

        for idx, (y, y_hat, emulator) in enumerate(izip(self.y.T, self.y_hat, self.emulators)):
            if self.method == 'gp':
                 out = emulator.predict(y, t, mean_only=not gp_errs)
                 if gp_errs:
                    mu, cov = out
                    all_cov.append(cov)
                 else:
                     mu = out
            else:
                mu = emulator.predict(t)
            # mu and cov come out as (1,) arrays.
            all_mu[:, idx] = mu + y_hat
            # all_err[:, idx] = np.sqrt(np.diag(cov))
            # all_cov[:, :, idx] = cov

        # Reshape to be consistent with my otehr implementation
        mu = all_mu.reshape((-1,))
        if not gp_errs:
            return mu

        cov = np.zeros((mu.shape[0], mu.shape[0]))
        nbins = self.y.shape[1]
        # This seems pretty inefficient; i'd like a more elegant way to do this.
        for n, c in enumerate(all_cov):
            for i, row in enumerate(c):
                for j, val in enumerate(row):
                    cov[i * nbins + n, j * nbins + n] = val
        return mu, cov

    def emulate_wrt_r(self, em_params, bin_centers, jackknife_errors=False, kind='slinear'):
        '''
        Conveniance function. Add's 'r' to the emulation automatically, as this is the
        most common use case.
        :param em_params:
            Dictionary of what values to predict at for each param. Values can be array
            or float.
        :param bin_centers:
            Centers of scale bins to predict at, for each point in HOD-space.
        :param kind:
            Kind of interpolation to do, is necessary. Default is slinear.
        :return:
            Mu and Cov, the predicted mu and covariance at em_params and bin_centers. If bin_centers
            is not equal to the bin_centers in the training data, the mean is interpolated as is the variance.
            Off diagonal elements are set to 0.
        '''
        # turns out this how it already works!
        mu, cov = self.emulate(em_params, jackknife_errors)
        # don't need to interpolate!
        if np.all(bin_centers == self.bin_centers):
            return mu, cov

        # TODO check bin_centers in bounds!
        # TODO is there any reasonable way to interpolate the covariance?
        all_err = np.sqrt(np.diag(cov))

        all_mu = mu.reshape((-1, len(self.bin_centers)))
        all_err = all_err.reshape((-1, len(self.bin_centers)))

        if len(all_mu.shape) == 1:  # just one calculation
            xi_interpolator = interp1d(self.bin_centers, all_mu, kind=kind)
            new_mu = xi_interpolator(bin_centers)
            err_interp = interp1d(self.bin_centers, all_err, kind=kind)
            new_err = err_interp(bin_centers)
            return new_mu, new_err

        new_mu, new_err = [], []
        for mean, err in izip(all_mu, all_err):
            xi_interpolator = interp1d(self.bin_centers, mean, kind=kind)
            interp_mean = xi_interpolator(bin_centers)
            new_mu.append(interp_mean)
            err_interp = interp1d(self.bin_centers, err, kind=kind)
            interp_err = err_interp(bin_centers)
            new_err.append(interp_err)
        mu = np.array(new_mu).reshape((-1,))
        cov = np.diag(np.array(new_err).reshape((-1,)))
        return mu, cov

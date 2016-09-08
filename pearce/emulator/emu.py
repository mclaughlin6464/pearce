#!/bin/bash
'''The Emu object esentially wraps the George gaussian process code. It handles building, training, and predicting.'''

from os import path
from time import time
import warnings
from glob import glob
from itertools import izip
import numpy as np
import scipy.optimize as op
from scipy.interpolate import interp1d
import george
from george.kernels import *

from .trainingData import PARAMS, GLOBAL_FILENAME
from .ioHelpers import global_file_reader, xi_file_reader


class Emu(object):
    # NOTE I believe that I'll have to subclass for the scale-indepent and original versions.
    # I'm going to write the original up so I can understand how I want this to work in particular, and may
    # have to copy a lot of it to the subclass

    # TODO load initial guesses from file.
    def __init__(self, training_dir, params=PARAMS, independent_variable='xi', fixed_params={}):

        '''
        # TODO change name of type
        self.type = type
        try:
            assert self.type in {'original_recipe', 'extra_crispy'}
        except AssertionError:
            raise AssertionError('Type %s not valid for Emu.'%self.type)
        '''
        if independent_variable == 'bias':
            raise NotImplementedError("I have to work on how to do xi_mm first.")

        assert independent_variable in {'xi', 'r2xi'}  # no bias for now.

        self.ordered_params = params

        self.get_training_data(training_dir, independent_variable, fixed_params)
        self.build_emulator(independent_variable, fixed_params)

    # I can get LHC from the training data. If any coordinate equals any other in its column we know!
    def get_training_data(self, training_dir, independent_variable, fixed_params):
        '''Implemented in subclasses. '''
        pass

    def build_emulator(self, independent_variable, fixed_params):
        '''
        Initialization of the emulator from recovered training data.
        :param independent_variable:
            independent_variable to emulate.
        :param fixed_params:
            Parameterst to hold fixed in teh training data
        :return: None
        '''
        ig = self.get_initial_guess(independent_variable, fixed_params)

        self.metric = []
        for p in self.ordered_params:
            if p.name in fixed_params:
                continue
            self.metric.append(ig[p.name])

        a = ig['amp']
        kernel = a * ExpSquaredKernel(self.metric, ndim=self.ndim)
        self.gp = george.GP(kernel)
        # gp = george.GP(kernel, solver=george.HODLRSolver, nleaf=x.shape[0]+1,tol=1e-18)
        self.fixed_params = fixed_params  # remember which params were fixed
        self.gp.compute(self.x, self.yerr)  # NOTE I'm using a modified version of george!

    # Not sure this will work at all in an LHC scheme.
    def get_plot_data(self, em_params, training_dir, independent_variable='xi', fixed_params={}):
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
        assert len(em_params) + len(fixed_params) == len(self.ordered_params)

        rbins, cosmo_params, method = global_file_reader(path.join(training_dir, GLOBAL_FILENAME))

        if method == 'LHC':
            raise ValueError('The data in training_dir is form a Latin Hypercube. \
                                        It is not possible to get plot data from this data. ')

        corr_files = sorted(glob(path.join(training_dir, 'xi*.npy')))
        cov_files = sorted(
            glob(path.join(training_dir, '*cov*.npy')))  # since they're sorted, they'll be paired up by params.
        rpoints = (rbins[:-1] + rbins[1:]) / 2
        nbins = rpoints.shape[0]

        npoints = len(corr_files)  # each file contains NBINS points in r, and each file is a 6-d point

        log_r = np.zeros((npoints, nbins))
        y = np.zeros((npoints, nbins))
        y_err = np.zeros((npoints, nbins))

        warned = False
        num_skipped = 0
        num_used = 0
        for idx, (corr_file, cov_file) in enumerate(izip(corr_files, cov_files)):
            params, xi, cov = xi_file_reader(corr_file, cov_file)

            # skip values that aren't where we've fixed them to be.
            # It'd be nice to do this before the file I/O. Not possible without putting all info in the filename.
            # or, a more nuanced file structure
            # TODO check if a fixed_param is not one of the options
            if any(params[key] != val for key, val in fixed_params.iteritems()):
                continue

            if any(params[key] != val for key, val in em_params.iteritems()):
                continue

            if np.any(np.isnan(cov)) or np.any(np.isnan(xi)):
                if not warned:
                    warnings.warn('WARNING: NaN detected. Skipping point in %s' % cov_file)
                    warned = True
                num_skipped += 1
                continue

            num_used += 1

            log_r[idx] = np.log10(rpoints)
            if independent_variable == 'xi':
                y[idx] = np.log10(xi)
                # Approximately true, may need to revisit
                # yerr[idx * NBINS:(idx + 1) * NBINS] = np.sqrt(np.diag(cov)) / (xi * np.log(10))
                y_err[idx] = np.sqrt(np.diag(cov)) / (
                    xi * np.log(10))  # I think this is right, extrapolating from the above.
            else:  # r2xi
                y[idx] = xi * rpoints * rpoints
                y_err[idx] = np.sqrt(np.diag(cov)) * rpoints  # I think this is right, extrapolating from the above.

        # remove rows that were skipped due to the fixed thing
        # NOTE: HACK
        # a reshape may be faster.
        zeros_slice = np.all(y != 0.0, axis=1)

        return log_r[zeros_slice], y[zeros_slice], y_err[zeros_slice]

    def get_initial_guess(self, independent_variable, fixed_params):
        '''
        Return the initial guess for the emulator, based on what the iv is. Guesses are learned from
        previous experiments.
        :param independent_variable:
            Which variable to return the guesses for.
        :param fixed_params:
            Parameters to hold fixed; only return guess for parameters that are not fixed.
        :return: initial_guesses, a dictionary of the guess for each parameter
        '''
        if independent_variable == 'xi':
            ig = {'amp': 0.481, 'logMmin': 0.1349, 'sigma_logM': 0.089,
                  'logM0': 2.0, 'logM1': 0.204, 'alpha': 0.039,
                  'f_c': 0.041, 'r': 0.040}
        else:  # independent_variable == 'r2xi':
            ig = {'amp': 1}
            ig.update({p.name: 0.1 for p in self.ordered_params})

        # remove entries for variables that are being held fixed.
        for key in fixed_params.iterkeys():
            del ig[key]

        return ig

    def train(self, **kwargs):
        pass

    def emulate(self, em_params):
        pass

    def emulate_wrt_r(self, em_params, rpoints):
        pass

    def goodness_of_fit(self, truth_dir, N=None, statistic='r2'):
        '''
        Calculate the goodness of fit of an emulator as compared to some validation data.
        :param truth_dir:
            Directory structured similary to the training data, but NOT used for training.
        :param N:
            Number of points to use to calculate G.O.F. measures. "None" tests against all values in truth_dir. If N
            is less than the number of points, N are randomly selected.
        :param statistic:
            What G.O.F. statistic to calculate. Default is R2. Other option is rmsfd.
        :return: values, a numpy arrray of the calculated statistics at each of the N training opints.
        '''
        assert statistic in {'r2', 'rmsfd'}

        corr_files = sorted(glob(path.join(truth_dir, '*corr*.npy')))
        cov_files = sorted(glob(path.join(truth_dir, '*cov*.npy')))

        np.random.seed(int(time()))

        if N is None:
            idxs = np.arange(len(corr_files))
        else:
            idxs = np.random.choice(len(corr_files), N, replace=False)

        values = []
        rbins, _, _ = global_file_reader(path.join(truth_dir, GLOBAL_FILENAME))
        r_centers = (rbins[:1] + rbins[:-1]) / 2
        for idx in idxs:
            params, true_xi, _ = xi_file_reader(corr_files[idx], cov_files[idx])
            pred_log_xi, _ = self.emulate_wrt_r(params, np.log10(r_centers))

            if statistic == 'rmsfd':
                values.append(np.sqrt(np.mean(((pred_log_xi - np.log10(true_xi)) ** 2) / (np.log10(true_xi) ** 2))))
            else:  # r2
                SSR = np.sum((pred_log_xi - np.log10(true_xi)) ** 2)
                SST = np.sum((np.log10(true_xi) - np.log10(true_xi).mean()) ** 2)

                values.append(1 - SSR / SST)

        return np.array(values)


class OriginalRecipe(Emu):
    def get_training_data(self, training_dir, independent_variable, fixed_params):
        '''
        Read the training data for the emulator and attach it to the object.
        :param training_dir:
            Directory where training data from trainginData is stored.
        :param independent_variable:
            Independant variable to emulate. Options are xi, r2xi, and bias (eventually).
        :param fixed_params:
            Parameters to hold fixed. Only available if data in training_dir is a full hypercube, not a latin hypercube.
        :return: None
        '''

        rbins, cosmo_params, method = global_file_reader(path.join(training_dir, GLOBAL_FILENAME))

        if not fixed_params and method == 'LHC':
            raise ValueError('Fixed parameters is not empty, but the data in training_dir is form a Latin Hypercube. \
                                Cannot performs slices on a LHC.')

        corr_files = sorted(glob(path.join(training_dir, 'xi*.npy')))
        cov_files = sorted(
            glob(path.join(training_dir, '*cov*.npy')))  # since they're sorted, they'll be paired up by params.
        self.rpoints = (rbins[:-1] + rbins[1:]) / 2
        nbins = self.rpoints.shape[0]
        # HERE
        npoints = len(corr_files) * nbins  # each file contains NBINS points in r, and each file is a 6-d point

        # HERE
        varied_params = set([p.name for p in self.ordered_params]) - set(fixed_params.keys())
        ndim = len(varied_params)  # lest we forget r

        # not sure about this.

        x = np.zeros((npoints, ndim))
        y = np.zeros((npoints,))
        yerr = np.zeros((npoints,))

        warned = False
        num_skipped = 0
        num_used = 0
        for idx, (corr_file, cov_file) in enumerate(izip(corr_files, cov_files)):
            params, xi, cov = xi_file_reader(corr_file, cov_file)

            # skip values that aren't where we've fixed them to be.
            # It'd be nice to do this before the file I/O. Not possible without putting all info in the filename.
            # or, a more nuanced file structure
            # TODO check if a fixed_param is not one of the options i.e. typo
            if any(params[key] != val for key, val in fixed_params.iteritems()):
                continue

            if np.any(np.isnan(cov)) or np.any(np.isnan(xi)):
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
                if p.name == 'r':
                    file_params.append(np.log10(self.rpoints))
                else:
                    file_params.append(np.ones((nbins,)) * params[p.name])

            x[idx * nbins:(idx + 1) * nbins, :] = np.stack(file_params).T
            # TODO helper function that handles this part
            # TODO the time has come to do something smarter for bias... I will ignore for now.
            '''
            if independent_variable == 'bias':
                y[idx * NBINS:(idx + 1) * NBINS] = xi / xi_mm
                ycovs.append(cov / np.outer(xi_mm, xi_mm))
            '''
            if independent_variable == 'xi':
                y[idx * nbins:(idx + 1) * nbins] = np.log10(xi)
                # Approximately true, may need to revisit
                yerr[idx * nbins:(idx + 1) * nbins] = np.sqrt(np.diag(cov)) / (xi * np.log(10))
                # ycovs.append(cov / (np.outer(xi, xi) * np.log(10) ** 2))  # I think this is right, extrapolating from the above.
            else:  # r2xi
                y[idx * nbins:(idx + 1) * nbins] = xi * self.rpoints * self.rpoints
                yerr[idx * nbins:(idx + 1) * nbins] = cov * np.outer(self.rpoints, self.rpoints)

        # ycov = block_diag(*ycovs)
        # ycov = np.sqrt(np.diag(ycov))

        # remove rows that were skipped due to the fixed thing
        # NOTE: HACK
        # a reshape may be faster.
        zeros_slice = np.all(x != 0.0, axis=1)
        # set the results of these calculations.
        self.ndim = ndim
        self.x = x[zeros_slice]
        self.y = y[zeros_slice]
        self.yerr = yerr[zeros_slice]

        self.y_hat = self.y.mean()
        self.y -= self.y_hat  # mean-subtract.

    def train(self, **kwargs):
        '''
            Train the emulator. Has a spotty record of working. Better luck may be had with the NAMEME code.
            :param kwargs:
                Kwargs that will be passed into the scipy.optimize.minimize
            :return: success: True if the training was successful.
            '''

        # move these outside? hm.
        def nll(p):
            # Update the kernel parameters and compute the likelihood.
            # params are log(a) and log(m)
            self.gp.kernel[:] = p
            ll = self.gp.lnlikelihood(self.y, quiet=True)

            # The scipy optimizer doesn't play well with infinities.
            return -ll if np.isfinite(ll) else 1e25

        # And the gradient of the objective function.
        def grad_nll(p):
            # Update the kernel parameters and compute the likelihood.
            self.gp.kernel[:] = p
            return -self.gp.grad_lnlikelihood(self.y, quiet=True)

        p0 = self.gp.kernel.vector
        results = op.minimize(nll, p0, jac=grad_nll, **kwargs)
        # results = op.minimize(nll, p0, jac=grad_nll, method='TNC', bounds =\
        #   [(np.log(0.01), np.log(10)) for i in xrange(ndim+1)],options={'maxiter':50})

        self.gp.kernel[:] = results.x
        self.gp.recompute()

        return results.success

    def emulate(self, em_params):
        '''
        Perform predictions with the emulator.
        :param varied_em_params:
            Dictionary of what values to predict at for each param. Values can be
            an array or a float.
        :return: mu, cov. The predicted value and the covariance matrix for the predictions
        '''
        input_params = {}
        input_params.update(self.fixed_params)
        input_params.update(em_params)
        assert len(input_params) == self.ndim  # check dimenstionality
        for i in input_params:
            assert any(i == p.name for p in self.ordered_params)

        # i'd like to remove 'r'. possibly requiring a passed in param?
        t_list = [input_params[p.name] for p in self.ordered_params]
        t_grid = np.meshgrid(*t_list)
        t = np.stack(t_grid).T
        t = t.reshape((-1, self.ndim))

        mu, cov = self.gp.predict(self.y, t)
        mu.reshape((-1, len(self.rpoints)))
        errs = np.sqrt(np.diag(cov))
        errs.reshape((-1, len(self.rpoints)))
        # Note ordering is unclear if em_params has more than 1 value.
        outputs = []
        for m, e in izip(mu, errs):
            outputs.append((m, e))
        return outputs

    def emulate_wrt_r(self, em_params, rpoints):
        '''
        Conveniance function. Add's 'r' to the emulation automatically, as this is the
        most common use case.
        :param em_params:
            Dictionary of what values to predict at for each param. Values can be array
            or float.
        :param rpoints:
            Points in 'r' to predict at, for each point in HOD-space.
        :return:
        '''
        vep = dict(em_params)
        vep.update({'r': rpoints})

        return self.emulate(vep)


class ExtraCrispy(Emu):
    def get_training_data(self, training_dir, independent_variable, fixed_params):
        '''
        Read the training data for the emulator and attach it to the object.
        :param training_dir:
            Directory where training data from trainginData is stored.
        :param independent_variable:
            Independant variable to emulate. Options are xi, r2xi, and bias (eventually).
        :param fixed_params:
            Parameters to hold fixed. Only available if data in training_dir is a full hypercube, not a latin hypercube.
        :return: None
        '''

        rbins, cosmo_params, method = global_file_reader(path.join(training_dir, GLOBAL_FILENAME))

        if not fixed_params and method == 'LHC':
            raise ValueError('Fixed parameters is not empty, but the data in training_dir is form a Latin Hypercube. \
                                Cannot performs slices on a LHC.')

        corr_files = sorted(glob(path.join(training_dir, 'xi*.npy')))
        cov_files = sorted(
            glob(path.join(training_dir, '*cov*.npy')))  # since they're sorted, they'll be paired up by params.
        self.rpoints = (rbins[:-1] + rbins[1:]) / 2
        nbins = self.rpoints.shape[0]
        # HERE
        npoints = len(corr_files)  # each file contains NBINS points in r, and each file is a 6-d point

        # HERE
        varied_params = set(self.ordered_params) - set(fixed_params.keys())
        ndim = len(varied_params)  # lest we forget r

        # not sure about this.

        x = np.zeros((npoints, ndim))
        y = np.zeros((npoints, nbins))
        #yerr = np.zeros((npoints, nbins))

        warned = False
        num_skipped = 0
        num_used = 0
        for idx, (corr_file, cov_file) in enumerate(izip(corr_files, cov_files)):
            params, xi, cov = xi_file_reader(corr_file, cov_file)

            # skip values that aren't where we've fixed them to be.
            # It'd be nice to do this before the file I/O. Not possible without putting all info in the filename.
            # or, a more nuanced file structure
            # TODO check if a fixed_param is not one of the options i.e. typo
            if any(params[key] != val for key, val in fixed_params.iteritems()):
                continue

            if np.any(np.isnan(cov)) or np.any(np.isnan(xi)):
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
                # HERE
                file_params.append(params[p.name])

            x[idx, :] = np.stack(file_params).T
            # TODO helper function that handles this part
            # TODO the time has come to do something smarter for bias... I will ignore for now.
            '''
            if independent_variable == 'bias':
                y[idx * NBINS:(idx + 1) * NBINS] = xi / xi_mm
                ycovs.append(cov / np.outer(xi_mm, xi_mm))
            '''
            if independent_variable == 'xi':
                y[idx, :] = np.log10(xi)
                # Approximately true, may need to revisit
                #yerr[idx, :] = np.sqrt(np.diag(cov)) / (xi * np.log(10))
                # ycovs.append(cov / (np.outer(xi, xi) * np.log(10) ** 2))  # I think this is right, extrapolating from the above.
            else:  # r2xi
                y[idx, :] = xi * self.rpoints * self.rpoints
                #yerr[idx, :] = cov * np.outer(self.rpoints, self.rpoints)

        # ycov = block_diag(*ycovs)
        # ycov = np.sqrt(np.diag(ycov))

        # remove rows that were skipped due to the fixed thing
        # NOTE: HACK
        # a reshape may be faster.
        zeros_slice = np.all(x != 0.0, axis=1)
        # set the results of these calculations.
        self.ndim = ndim
        self.x = x[zeros_slice]
        self.y = y[zeros_slice, :]
        self.yerr = np.zeros((self.x.shape[0], )) #We don't use errors for extra crispy!
        #self.yerr = yerr[zeros_slice]

        self.y_hat = self.y.mean(axis=0)
        self.y -= self.y_hat  # mean-subtract.

    def train(self, **kwargs):
        '''
        Train the emulator. Has a spotty record of working. Better luck may be had with the NAMEME code.
        :param kwargs:
            Kwargs that will be passed into the scipy.optimize.minimize
        :return: success: True if the training was successful.
        '''

        # move these outside? hm.
        def nll(p):
            # Update the kernel parameters and compute the likelihood.
            # params are log(a) and log(m)
            self.gp.kernel[:] = p
            # check this has the right direction
            ll = np.sum(self.gp.lnlikelihood(y, quiet=True) for y in self.y)

            # The scipy optimizer doesn't play well with infinities.
            return -ll if np.isfinite(ll) else 1e25

        # And the gradient of the objective function.
        def grad_nll(p):
            # Update the kernel parameters and compute the likelihood.
            self.gp.kernel[:] = p
            # mean or sum?
            return -np.mean(self.gp.grad_lnlikelihood(y, quiet=True) for y in self.y)

        p0 = self.gp.kernel.vector
        results = op.minimize(nll, p0, jac=grad_nll, **kwargs)
        # results = op.minimize(nll, p0, jac=grad_nll, method='TNC', bounds =\
        #   [(np.log(0.01), np.log(10)) for i in xrange(ndim+1)],options={'maxiter':50})

        self.gp.kernel[:] = results.x
        self.gp.recompute()

        return results.success

    def emulate(self, em_params):
        '''
        Perform predictions with the emulator.
        :param varied_em_params:
            Dictionary of what values to predict at for each param. Values can be
            an array or a float.
        :return: mu, cov. The predicted value and the covariance matrix for the predictions
        '''
        input_params = {}
        input_params.update(self.fixed_params)
        input_params.update(em_params)
        assert len(input_params) == self.ndim  # check dimenstionality
        for i in input_params:
            assert any(i == p.name for p in self.ordered_params)

        # i'd like to remove 'r'. possibly requiring a passed in param?
        t_list = [input_params[p.name] for p in self.ordered_params]
        t_grid = np.meshgrid(*t_list)
        t = np.stack(t_grid).T
        t = t.reshape((-1, self.ndim))

        output = []
        for y, y_hat in zip(self.y.T, self.y_hat):
            mu, cov = self.gp.predict(y, t)
            print type(mu+y_hat), type(mu)
            output.append((mu + y_hat, np.sqrt(np.diag(cov))))
        # note may want to do a reshape here., Esp to be consistent with the other
        # implementation
        return output

    def emulate_wrt_r(self, em_params, rpoints, kind='slinear'):
        '''
        Conveniance function. Add's 'r' to the emulation automatically, as this is the
        most common use case.
        :param em_params:
            Dictionary of what values to predict at for each param. Values can be array
            or float.
        :param rpoints:
            Points in 'r' to predict at, for each point in HOD-space.
        :param kind:
            Kind of interpolation to do, is necessary. Default is slinear.
        :return:
        '''
        # turns out this how it already works!
        output = self.emulate(em_params)
        # don't need to interpolate!
        if np.all(rpoints == self.rpoints):
            return output
        # TODO check rpoints in bounds!
        new_output = []
        for mean, err in output:
            xi_interpolator = interp1d(rpoints, mean, kind=kind)
            interp_mean = xi_interpolator(rpoints)
            print type(interp_mean)
            err_interp = interp1d(rpoints, err, kind=kind)
            interp_err = err_interp(rpoints)
            new_output.append((interp_mean, interp_err))
        return new_output

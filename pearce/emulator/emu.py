#!/bin/bash
'''The Emu object esentially wraps the George gaussian process code. It handles building, training, and predicting.'''

from os import path
import warnings
from glob import glob
from itertools import izip
import numpy as np
import scipy.optimize as op
import george
from george.kernels import *

from .trainingData import PARAMS, GLOBAL_FILENAME
from .ioHelpers import global_file_reader, xi_file_reader

class Emu(object):

    #NOTE I believe that I'll have to subclass for the scale-indepent and original versions.
    #I'm going to write the original up so I can understand how I want this to work in particular, and may
    # have to copy a lot of it to the subclass

    #TODO load initial guesses from file.
    def __init__(self, training_dir, independent_variable='xi',fixed_params = {}):

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

        assert independent_variable in {'xi', 'r2xi'} #no bias for now.

        self.get_training_data(self, training_dir, independent_variable, fixed_params)
        ig = self.get_initial_guess(independent_variable)

        self.metric = []
        for p in PARAMS:
            if p.name in fixed_params:
                continue
            self.metric.append(ig[p.name])

        self.metric.append(ig['r'])

        a = ig['amp']
        kernel = a * ExpSquaredKernel(self.metric, ndim=self.ndim)
        self.gp = george.GP(kernel)
        # gp = george.GP(kernel, solver=george.HODLRSolver, nleaf=x.shape[0]+1,tol=1e-18)

        self.gp.compute(self.x, self.yerr)  # NOTE I'm using a modified version of george!

    # I can get LHC from the training data. If any coordinate equals any other in its column we know!
    def get_training_data(self,training_dir, independent_variable, fixed_params):
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

        if not fixed_params and method=='LHC':
            raise ValueError('Fixed parameters is not empty, but the data in training_dir is form a Latin Hypercube. \
                                Cannot performs slices on a LHC.')

        corr_files = sorted(glob(path.join(training_dir, 'xi*.npy')))
        cov_files = sorted(
            glob(path.join(training_dir, '*cov*.npy')))  # since they're sorted, they'll be paired up by params.
        self.rpoints = (rbins[:-1]+rbins[1:])/2
        nbins = self.rpoints.shape[0]
        #HERE
        npoints = len(corr_files) * nbins # each file contains NBINS points in r, and each file is a 6-d point

        #HERE
        varied_params = set(PARAMS) - set(fixed_params.keys())
        ndim = len(varied_params) + 1  # lest we forget r

        #not sure about this.

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
            for p in PARAMS:
                if p.name in fixed_params:
                    continue
                #HERE
                file_params.append(np.ones((nbins,)) * params[p.name])

            file_params.append(np.log10(self.rpoints))

            x[idx * nbins:(idx + 1) * nbins, :] = np.stack(file_params).T
            #TODO helper function that handles this part
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
                #ycovs.append(cov / (np.outer(xi, xi) * np.log(10) ** 2))  # I think this is right, extrapolating from the above.
            else: #r2xi
                y[idx * nbins:(idx + 1) * nbins] = xi * self.rpoints * self.rpoints
                yerr[idx * nbins:(idx + 1) * nbins] = cov * np.outer(self.rpoints, self.rpoints)

        # ycov = block_diag(*ycovs)
        # ycov = np.sqrt(np.diag(ycov))

        # remove rows that were skipped due to the fixed thing
        # NOTE: HACK
        # a reshape may be faster.
        zeros_slice = np.all(x != 0.0, axis=1)
        #set the results of these calculations.
        self.ndim = ndim
        self.x =  x[zeros_slice]
        self.y = y[zeros_slice]
        self.yerr = yerr[zeros_slice]

        self.y_hat = self.y.mean()
        self.y-=self.y_hat #mean-subtract.

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
        assert len(em_params) + len(fixed_params) + 1 == len(PARAMS)

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
            ig = {'amp': 0.481, 'logMmin':0.1349,'sigma_logM':0.089,
                   'logM0': 2.0, 'logM1':0.204, 'alpha':0.039,
                   'f_c':0.041, 'r':0.040}
        else:# independent_variable == 'r2xi':
            ig = {'amp':1}
            ig.update({p.name:0.1 for p in PARAMS})

        #remove entries for variables that are being held fixed.
        for key in fixed_params.iterkeys():
            del ig[key]

        return ig

    def train(self, **kwargs):
        '''
        Train the emulator. Has a spotty record of working. Better luck may be had with the NAMEME code.
        :param kwargs:
            Kwargs that will be passed into the scipy.optimize.minimize
        :return: success: True if the training was successful.
        '''
        #move these outside? hm.
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
            return -self.gp.grad_lnlikelihood(y, quiet=True)

        p0 = self.gp.kernel.vector
        results = op.minimize(nll, p0, jac=grad_nll, **kwargs)
        #results = op.minimize(nll, p0, jac=grad_nll, method='TNC', bounds =\
        #   [(np.log(0.01), np.log(10)) for i in xrange(ndim+1)],options={'maxiter':50})

        self.gp.kernel[:] = results.x
        self.gp.recompute()

        return results.success

    def emulate(self, fixed_em_params, varied_em_params):

        input_params = set(fixed_em_params) | set(varied_em_params)
        assert len(input_params) == self.ndim  # check dimenstionality
        for i in input_params:
            assert any(i==p.name for p in PARAMS or i=='r')

        if y_param is None:
            # We only have to vary one parameter, as given by x_points
            t_list = []
            for p in PARAMS:
                if p in em_params:
                    t_list.append(np.ones_like(x_points) * em_params[p])
                elif p == x_param:
                    t_list.append(x_points)
                else:
                    continue
            # adding 'r' in as a special case
            if 'r' in em_params:
                t_list.append(np.ones_like(x_points) * em_params['r'])
            elif 'r' == x_param:
                t_list.append(x_points)

            t = np.stack(t_list).T

            # TODO mean subtraction?
            mu, cov = gp.predict(em_y, t)

            # TODO return std or cov?
            # TODO return r's too? Just have those be passed in?
            em_y+=em_y_hat
            return mu+em_y_hat, np.diag(cov)
        else:
            output = []
            assert len(y_points) <= 20  # y_points has a limit, otherwise this'd be crazy


            for y in y_points:  # I thought this had a little too mcuh copying, but
                # this is the best wayt ensure the ordering is consistent.
                t_list = []
                for p in PARAMS:
                    if p in em_params:
                        t_list.append(np.ones_like(x_points) * em_params[p])
                    elif p == x_param:
                        t_list.append(x_points)
                    elif p == y_param:
                        t_list.append(np.ones_like(x_points) * y)
                    else:
                        continue

                if 'r' in em_params:
                    t_list.append(np.ones_like(x_points) * em_params['r'])
                elif 'r' == x_param:
                    t_list.append(x_points)
                elif 'r' == y_param:
                    t_list.append(np.ones_like(x_points) * y)

                t = np.stack(t_list).T

                mu, cov = gp.predict(em_y, t)
                output.append((mu+em_y_hat, np.sqrt(np.diag(cov))))
            em_y+=em_y_hat
            return output
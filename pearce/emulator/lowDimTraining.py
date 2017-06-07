#!/bin/bash
'''Training in high dimensions is difficult. This module incorporates functions that learn
hyperparamters from lower dimensions of data.'''
from os import path
from glob import glob
from itertools import combinations, product, izip
from collections import defaultdict, OrderedDict

import numpy as np

from .emu import OriginalRecipe, ExtraCrispy
from .ioHelpers import obs_file_reader

#TODO does this need to be in a separate file here? Can I attach it to emu?
def low_dim_train(training_dir,emu_type = 'OriginalRecipe', emu_kwargs = {}):
    '''
    Train the hyperparameters in lower dimensions and average them together.
    :param training_dir:
        Directory with the training data, passed into the emu object
    :param ordered_params:
        Ordered parameter list, passed into the emu object
    :param independent_variable:
        The independent variable to emulate and learn params for.
    :param n_params:
        Number of parameters to hold train the emulator with. Default is 3.
    :param emu_type:
        Type of emulator to use. Default is OriginalRecipe
    :return:
    '''

    assert emu_type in {'OriginalRecipe', 'ExtraCrispy'}
    if emu_type == 'ExtraCrispy':
        if not emu_kwargs: # need to speciy default params
            emu_kwargs = {'experts':10, 'overlap': 1}
    emu_obj = OriginalRecipe if emu_type == 'OriginalRecipe' else ExtraCrispy

    # TODO read this from training_dir
    cosmologies = np.loadtxt(glob(path.join(training_dir, 'cosmology*.dat'))[0])
    HODs = np.loadtxt(glob(path.join(training_dir, 'HOD*.dat'))[0])

    cosmo_params = ['Omega_m', 'Omega_b', 'sigma_8', 'h', 'n_s', 'N_eff', 'w_de']
    HOD_params = ['Msat', 'alpha', 'Mcut', 'sigma_logM', 'vbias_cen', 'vbias_sats', 'conc', 'v_field_amp']

    op_names = HOD_params[:]
    op_names.extend(cosmo_params)
    min_max_vals = zip(np.r_[HODs.min(axis=0), cosmologies.min(axis=0)], \
                       np.r_[HODs.max(axis=0), cosmologies.max(axis=0)])
    ordered_params = OrderedDict(izip(op_names, min_max_vals))
    ordered_params['r'] = (0,1)

    hyper_params = {p: [] for p in ordered_params}
    hyper_params['amp'] = [] #special case
    #unique values in the training data

    #optimize HOD params by holding cosmo fixed. Then, do the opposite for cosmo and HOD.

    for fixed_key,fixed_values, varied_values, varied_params  in izip( ['cosmo', 'HOD'],(cosmologies, HODs), \
                                                                       (HODs, cosmologies),  (HOD_params, cosmo_params)):
        for fno in xrange(fixed_values.shape[0]):

            if 'fixed_params' in emu_kwargs:
                emu_kwargs['fixed_params'].update({fixed_key: fno})
                emu = emu_obj(training_dir,  **emu_kwargs)
            else:
                fixed_params = {fixed_key: fno}
                emu = emu_obj(training_dir, fixed_params = fixed_params, **emu_kwargs)
            success = emu.train()

        if not success:
            continue

        for p, m in zip(varied_params, emu.metric[1:]):
            hyper_params[p.name].append(m)
        hyper_params['amp'].append(emu.metric[0])

    for key in hyper_params:
        hyper_params[key] = np.array(hyper_params[key])

    for key, val in hyper_params.iteritems():
        print key,val.shape,np.median(val), val.mean(), val.std()
    print

    return {key:val.mean() for key, val in hyper_params.iteritems()}

    '''
    for pc in param_combinations:
        #for each combination, also train for each combination of unique values
        #we're being very thorough
        #these are unique values of the params were holding fixed, those not in pc
        fixed_pc = [p for p in varied_params if p not in pc]
        unique_values_pc = product(*[list(unique_values[p.name]) for p in fixed_pc])
        n_uv = np.prod([len(list(unique_values[p.name])) for p in fixed_pc])

        if n_uv > n_max:
            #select a subsample
            unique_values_idx = set(np.random.choice(n_uv,size = n_max, replace=False))
        else:
            unique_values_idx = set(xrange(n_uv))

        #now, rebuild and train the emulator.
        for idx, uv in enumerate(unique_values_pc):
            if idx not in unique_values_idx:
                continue
            print uv
            #hold these parameters fixed
            fixed_params = {p.name:uv[idx] for idx, p in enumerate(fixed_pc)}
            if emu is None:
                emu = emu_obj(training_dir, ordered_params, independent_variable, fixed_params)
            else:
                emu.get_training_data(training_dir, independent_variable, fixed_params)
                emu.build_emulator(independent_variable, fixed_params)
            success = emu.train()

            if not success:
                continue

            for p, m in zip(pc, emu.metric[1:]):
                hyper_params[p.name].append(m)
            hyper_params['amp'].append(emu.metric[0])
            if ordered_params[-1].name == 'r': #has 'r'
                hyper_params['r'].append(emu.metric[-1])
        print

    for key in hyper_params:
        hyper_params[key] = np.array(hyper_params[key])

    for key, val in hyper_params.iteritems():
        print key,val.shape,np.median(val), val.mean(), val.std()
    print

    return {key:val.mean() for key, val in hyper_params.iteritems()}
    '''
def get_unique_values(training_dir):
    '''
    Returns a dictionary of sets containing the unique training values of each
    parameter in the training_dir. I tried to wrap this feature into the emu
    object. However, that requires loading all data at once, something that isn't
    always possible and led to the creation of this module in the first place.
    :param training_dir:
        Directory holding the training data for the emulator.
    :return:
        unique_values, a dictionary of sets containins unique values of
        the parameters in the training file.
    '''
    unique_values = defaultdict(set)
    obs_files = glob(path.join(training_dir, 'obs*.npy'))
    for obs_file in obs_files:
        params, _ = obs_file_reader(obs_file)
        for key, val in params.iteritems():
            unique_values[key].add(val)

    return unique_values

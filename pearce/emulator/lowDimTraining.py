#!/bin/bash
'''Training in high dimensions is difficult. This module incorporates functions that learn
hyperparamters from lower dimensions of data.'''
from os import path
from glob import glob
from itertools import combinations, product
from collections import defaultdict

import numpy as np

from .emu import OriginalRecipe, ExtraCrispy
from .ioHelpers import training_file_loc_reader, params_file_reader, global_file_reader

#TODO does this need to be in a separate file here? Can I attach it to emu?
def low_dim_train(training_dir, ordered_params, independent_variable, n_params = 3, emu_type = 'OriginalRecipe'):
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

    emu_obj = OriginalRecipe if emu_type == 'OriginalRecipe' else ExtraCrispy

    varied_params = ordered_params.keys()
    if 'r' in ordered_params:
        #we don' do combinations in 'r'
        varied_params.remove('r')
    if 'z' in ordered_params:
        varied_params.remove('z') 

    hyper_params = {p: [] for p in ordered_params}
    hyper_params['amp'] = [] #special case
    #unique values in the training data
    unique_values = get_unique_values(training_dir)#{p.name:list(set(emu.x[:, idx])) for idx, p in enumerate(varied_params)}
    assert len(unique_values)>=0
    #all unique combinations to train
    param_combinations = combinations(varied_params, n_params)

    emu = None
    n_max = 20#

    print len(param_combinations)*n_max

    for pc in param_combinations:
        #for each combination, also train for each combination of unique values
        #we're being very thorough
        #these are unique values of the params were holding fixed, those not in pc
        fixed_pc = [p for p in varied_params if p not in pc]
        unique_values_pc = product(*[list(unique_values[p]) for p in fixed_pc])
        n_uv = np.prod([len(list(unique_values[p])) for p in fixed_pc])

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
            fixed_params = {p:uv[idx] for idx, p in enumerate(fixed_pc)}
            if emu is None:
                emu = emu_obj(training_dir,independent_variable=independent_variable, fixed_params=fixed_params)
            else:
                emu.load_training_data(training_dir)
                emu.build_emulator({})
            success = emu.train_metric()

            if not success:
                continue

            for p, m in zip(pc, emu.metric[1:]):
                hyper_params[p].append(m)
            hyper_params['amp'].append(emu.metric[0])
            if 'r' in ordered_params: #has 'r'
                hyper_params['r'].append(emu.metric[-1])
        print '*'

    for key in hyper_params:
        hyper_params[key] = np.array(hyper_params[key])

    for key, val in hyper_params.iteritems():
        print key,val.shape,np.median(val), val.mean(), val.std()
    print
    output = dict()
    for key, val in hyper_params.iteritems():
        if len(val)==0:
            output[key] = 'None'
        else:
            output[key] = val.mean()
    return output#{key:val.mean() for key, val in hyper_params.iteritems()}

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
    scale_factor_dirs = glob(path.join(training_dir, '*/'))
    for sfd in scale_factor_dirs:
        #dict where each key is a tuple of params in the file
        tfl = training_file_loc_reader(sfd)
        ordered_params = params_file_reader(sfd)
        pnames = ordered_params.keys()
        for key in tfl:
            for p, val in zip(pnames, key):
                unique_values[p].add(val)
        
        #bins, _, _, _ = global_file_reader(sfd)
        #rbc = (bins[1:]+bins[:-1])/2.0
        #for r in rbc:
        #    unique_values['r'].add(r)
        #a = float(sfd.split('/')[-1][2:])
        #z = 1.0/a-1
        #unique_values['z'].add(z)

    return unique_values

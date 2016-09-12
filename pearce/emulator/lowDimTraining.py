#!/bin/bash
'''Training in high dimensions is difficult. This module incorporates functions that learn
hyperparamters from lower dimensions of data.'''
from itertools import combinations, product
import numpy as np
from .emu import OriginalRecipe, ExtraCrispy

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
    emu = emu_obj(training_dir, ordered_params, independent_variable)

    if ordered_params[-1].name == 'r':
        #we don' do combinations in 'r'
        varied_params = ordered_params[:-1]
    else:
        varied_params = ordered_params[:]
    #TODO need to add amplitude and r
    hyper_params = {p.name: [] for p in ordered_params}
    hyper_params['amp'] = [] #special case
    #unique values in the training data
    unique_values = {p.name:list(set(emu.x[:, idx])) for idx, p in enumerate(varied_params)}
    #all unique combinations to train
    param_combinations = combinations(varied_params, n_params)

    for pc in param_combinations:
        print pc.name
        #for each combination, also train for each combination of unique values
        #we're being very thorough
        #these are unique values of the params were holding fixed, those not in pc
        fixed_pc = [p for p in varied_params if p not in pc]
        unique_values_pc = product(*[unique_values[p.name] for p in fixed_pc])

        #now, rebuild and train the emulator.
        for uv in unique_values_pc:
            print uv
            #hold these parameters fixed
            fixed_params = {p.name:uv[idx] for idx, p in enumerate(fixed_pc)}
            emu.get_training_data(training_dir, independent_variable, fixed_params)
            print emu.x.shape
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
        print key, val.mean(), val.std()

    return {key:val.mean() for key, val in hyper_params.iteritems()}

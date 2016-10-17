#!/bin/bash
from pearce.emulator.lowDimTraining import low_dim_train
from pearce.emulator.trainingData import PARAMS, parameter
training_dir = '/u/ki/swmclau2/des/Pearce_wp_FHC/'

or_params = PARAMS[:]
or_params.append(parameter('r', 0, 1)) #95% sure bounds aren't used for r
n_params = 3
metric = low_dim_train(training_dir, or_params, None, n_params)
for key, val in metric.iteritems():
    print key, val

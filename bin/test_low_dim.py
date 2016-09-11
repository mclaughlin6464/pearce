#!/bin/bash
from pearce.emulator.lowDimTraining import low_dim_train
from pearce.emulator.trainingData import PARAMS, parameter
training_dir = '/u/ki/swmclau2/des/PearceTraining/'

or_params = PARAMS[:]
or_params.append(parameter('r', 0, 1)) #95% sure bounds aren't used for r
independent_variable = 'xi'
n_params = 1
metric = low_dim_train(training_dir, or_params, independent_variable, n_params)
print metric
#!/bin/bash
from pearce.emulator.lowDimTraining import low_dim_train
training_dir = '/u/ki/swmclau2/des/PearceLHC_wp_z/'
from collections import OrderedDict
n_params = 3
metric = low_dim_train(training_dir, None, n_params)
for key, val in metric.iteritems():
    print key, val

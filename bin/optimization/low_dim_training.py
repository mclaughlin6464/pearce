#!/bin/bash
from pearce.emulator.lowDimTraining import low_dim_train
training_dir = '/u/ki/swmclau2/des/PearceFHC_wp_z/'
from collections import OrderedDict
or_params = OrderedDict([('logMmin', (11.7, 12.5)), ('sigma_logM', (0.2, 0.7)), ('logM0', (10, 13)),('logM1', (13.1, 14.3)),
                        ('alpha', (0.75, 1.25)),('f_c', (0.1, 0.5)), ('r', (0.093735900000000011, 34.082921444999997)),
                                                                                               ('z', (0.0, 0.5))])
n_params = 3
metric = low_dim_train(training_dir, or_params, None, n_params)
for key, val in metric.iteritems():
    print key, val

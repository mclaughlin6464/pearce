#!/bin/bash
'''Training helper. Is called from training data as main, to send to cluster.'''

import numpy as np
#load params, rbins, cosmo info in param file?
#create cat object
#iterate through params, saving results.

def loadParams(paramFile):
    '''
    Load the parameters to calculate. Will load bins, parameters, cosmology info, and output directory!
    :param paramFile:
        The file where the parameters are stored. The output directory is the same directory as the paramfile, and
        the radial bins are stored in the same directory as well.
    :return: rbins, parameters, cosmology info, output directory.
    '''
    params = np.loadtxt(paramFile)
    cosmo_params = {}
    with open(paramFile) as f:
        for i, line in enumerate(f):
            if line[0] != '#':
                continue  # only looking at comments, and first two lines don't have params. Note: Does have cosmo!
            splitLine = line.strip('# \n').split(':')  # split into key val pair
            cosmo_params[splitLine[0]] = float(splitLine[1])


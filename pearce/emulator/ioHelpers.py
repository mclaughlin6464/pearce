#!/bin/bash
'''Contains several helper functions for emulator file IO. For most part, these are very general
and useful in multiple cases. More specific, one-use functions are generally left where they are.'''
# TODO FOR GOD SAKES DECIDE ON CAMELCASE V UNDERSCORES YOU MADMAN

from os import path
import numpy as np


# TODO change to obs in title
def xi_file_reader(corr_file, cov_file=None):
    '''
    A helper function to parse the training data files.
    :param corr_file:
        Filename of the file with xi information
    :param cov_file:
        Optional. Filename containing the jackknifed covariance matrix.
    :return:
        HOD parameters, xi, cov (if cov_file is not None)
    '''

    assert path.exists(corr_file)
    # TODO change to obs
    xi = np.loadtxt(corr_file)  # not sure if this will work, might nead to transpose
    params = {}
    with open(corr_file) as f:
        for i, line in enumerate(f):
            # TODO skip an additional line, added obs.
            # TODO read observable.
            if line[0] != '#' or i < 2:
                continue  # only looking at comments, and first two lines don't have params. Note: Does have cosmo!
            splitLine = line.strip('# \n').split(':')  # split into key val pair
            params[splitLine[0]] = float(splitLine[1])

    if cov_file is not None:
        assert path.exists(cov_file)
        cov = np.loadtxt(cov_file)

        return params, xi, cov
    return params, xi


# TODO change RBINS to bins
def global_file_reader(global_filename):
    '''
    Helper function, useful for reading the information in the global file.
    :param global_filename:
        Path+filename for the global file.
    :return:
        rbins, cosmo_params
    '''
    rbins = np.loadtxt(global_filename)
    # cosmology parameters are stored in the global header
    cosmo_params = {}
    with open(global_filename) as f:
        for i, line in enumerate(f):
            if i == 0:
                splitLine = line.strip('# \n').split(':')  # split into key val pair
                method = splitLine[1]
            elif line[0] != '#' or i < 2:
                continue  # only looking at comments, and first two lines don't have params. Note: Does have cosmo!
            splitLine = line.strip('# \n').split(':')  # split into key val pair
            try:
                cosmo_params[splitLine[0]] = float(splitLine[1])
            except ValueError:
                cosmo_params[splitLine[0]] = splitLine[1]

    return rbins, cosmo_params, method


# Could use ConfigParser maybe
def config_reader(filename):
    '''
    General helper module. Turns a file of key:value pairs into a dictionary.
    :param filename:
        Config filename.
    :return:
        A dictionary of key-value pairs.
    '''
    config = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            splitline = line.split(':')
            config[splitline[0].strip()] = splitline[1].strip()

    return config

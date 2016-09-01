#!/bin/bash
'''This file samples points in the parameter space, and sends off jobs to perform the calculation of xi at those
points in parameter space. '''

from time import time
from collections import namedtuple
import numpy as np

parameter = namedtuple('Parameter', ['name', 'low','high'])

#global object that defines both the names and ordering of the parameters, as well as their boundaries.
#TODO consider moving?
#TODO reading in bounds/params from config
PARAMS = [ parameter('logMmin', 11.7,12.5),
           parameter('sigma_logM', 0.2, 0.7),
           parameter('logM0', 10,13),
           parameter('logM1', 13.1, 14.3),
           parameter('alpha', 0.75, 1.25),
           parameter('f_c', 0.1, 0.5)]
#I think that it's better to have this param global, as it prevents there from being any conflicts.

#make LHC
#make paramcube
#make kils command
#make sherlock command
#send commands to cluster

def makeLHC(N=500):
    '''Return a vector of points in parameter space that defines a latin hypercube.
    :param N:
        Number of points per dimension in the hypercube. Default is 500.
    :return
        A latin hyper cube sample in HOD space in a numpy array.
    '''
    np.random.seed(int(time()))

    #this is a bad name...
    ranges = []
    #by linspacing each parameter and shuffling, I ensure there is only one point in each row, in each dimension.
    for p in PARAMS:
        point = np.linspace(p.low, p.high, num=N)
        np.random.shuffle(point)#makes the cube random.
        ranges.append(point)
    return np.stack(ranges).T

def makeFHC(N=4):
    '''
    Return a vector of points in parameter space that defines a afull hyper cube.
    :param N:
        Number of points per dimension. Can be an integer or list. If it's a number, it will be the same
        across each dimension. If a list, defines points per dimension in the same ordering as PARAMS.
    :return:
        A full hyper cube sample in HOD space in a numpy array.
    '''

    if type(N) is int:
        N = [N for i in xrange(len(PARAMS))]

    assert type(N) is list
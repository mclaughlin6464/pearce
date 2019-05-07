from os import path
from time import time
import warnings
from itertools import product
import yaml
import numpy as np
import pandas as pd
from mpi4py import MPI
import h5py

from pearce.mocks import cat_dict

from sys import argv
from sys import stdout

if __name__ == '__main__':

    comm = MPI.COMM_WORLD

    rank = comm.Get_rank()
    size = comm.Get_size()
    print rank, MPI.Get_processor_name()
    stdout.flush()


'''
Implementing an algorithm for VPF designed by Tom Abel and
Arka Banerjee  for computing the CDF of void volumes.

Making it halotools compliant will be annoying...
'''
import numpy as np
from scipy.spatial import cKDTree
from time import time

def void_density_function(sample1, rbins, n_ran=1e9, randoms=None,
                         period = None, seed = None, leafsize=16, eps=0.0, n_jobs = 16, PBC=True):
    """
    Compute the CDF of void sizes for the sample
    :param sample1:
        Nx3 positions of the sample
    :param rbins:
        scale bins to prode void CDFs at
    :param n_ran:
        Number of random points. Must specify this or randoms
    :param randoms:
        Random points. Must be specified or n_ran
    :param period:
        Period of the box, default is None. Must be specified with n_ran
    :param seed:
        random seed, default is None
    :param leafsize:
        Size of leaves for KDtree. Default is 16
    :param eps:
        Size of approximation for KDtree query. Default is 0.0, to return exact NNs
    :return:
        VDF, the cdf of the VPF
    """
    return knn_cdf(sample1, rbins, k=1, n_ran=n_ran, randoms=randoms,
                         period = period, seed = seed, leafsize=leafsize, eps=eps, n_jobs = n_jobs,
                         PBC=PBC)

def knn_cdf(sample1, rbins, k, n_ran=1e9, randoms=None,
                         period = None, seed = None, leafsize=16, eps=0.0, n_jobs = 16, PBC=True):
    """
    Compute the CDF of void sizes for the sample
    :param sample1:
        Nx3 positions of the sample
    :param rbins:
        scale bins to prode void CDFs at
    :param k:
        int k-th nearest neighbor to query
    :param n_ran:
        Number of random points. Must specify this or randoms
    :param randoms:
        Random points. Must be specified or n_ran
    :param period:
        Period of the box, default is None. Must be specified with n_ran
    :param seed:
        random seed, default is None
    :param leafsize:
        Size of leaves for KDtree. Default is 16
    :param eps:
        Size of approximation for KDtree query. Default is 0.0, to return exact NNs
    :return:
        VDF, the cdf of the VPF
    """
    assert sample1.shape[1] == 3, "Invalid sample1 shape"
    assert type(k) is int

    if seed is None:
        seed = int(time())
    np.random.seed(seed)

    if randoms is None and n_ran is None:
        raise AssertionError("must specify either randoms or n_ran")
    elif randoms is None: # n_ran is not None
        assert period is not None
        randoms = np.random.rand(int(n_ran), 3)*period

    # i think theres a weird boundary condition error here
    boxsize = period+1e-6 if PBC else None 
    tree = cKDTree(sample1, boxsize=boxsize, leafsize=leafsize)

    void_size = tree.query(randoms, k=[k], eps=eps, n_jobs=n_jobs)[0].squeeze()
    bin_centers = (rbins[1:]+rbins[:-1])/2.0

    sorted_void_size = np.sort(void_size)
    # possibly more efficiency gains if we can assume bin_centers is sorted
    # however were mainly limited by the tree stuff so probably doesn't matter.
    unnorm_CDF = np.searchsorted(sorted_void_size, bin_centers)
    # TODO could add option to return the tree, and pass it in.
    return unnorm_CDF*1.0/sorted_void_size.shape[0]

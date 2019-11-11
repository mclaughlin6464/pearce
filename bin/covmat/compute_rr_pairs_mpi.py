from halotools.mock_observables.pair_counters import npairs_jackknife_3d
from halotools.mock_observables.catalog_analysis_helpers import cuboid_subvolume_labels
from pearce.mocks.kittens import TestBox
import yaml
import numpy as np
from itertools import product

n_split = 20
n_sub = 5
n_rands = 5
n_cores = 16

def run(comm):

    rank, size = comm.Get_rank(), comm.Get_size()
    
    if rank ==0:
        config_fname = 'xi_cosmo_trainer.yaml'

        with open(config_fname, 'r') as ymlfile:
            cfg = yaml.load(ymlfile)

        min_ptcl = int(cfg['HOD']['min_ptcl']) 
        r_bins = np.array(cfg['observation']['bins'] ).astype(float)
        cat = TestBox(boxno = 0, realization = 0, system = 'ki-ls')
        cat.load(1.0, HOD = str('zheng07'), particles = True, downsample_factor = 1e-2)

        lbox_dict = {"Lbox": cat.Lbox/cat.h} # too lazy to figure out how to broadcast the float

        n_cores = 16
        n_cores = cat._check_cores(n_cores)
        rbins = np.array(rbins)

        randoms = np.random.random((int(cat.halocat.ptcl_table['x'].shape[0] * n_rands), 3))*cat.Lbox/cat.h  # Solution to NaNs: Just fuck me up with randoms
        np.savetxt('randoms_mpi.npy', randoms)

        split_randoms = np.array_split(randoms, n_split, axis = 0)

        n_combos = n_split*(n_split+1)/2.0
        n_per_node = int(np.ceil(n_combos/size))

        sendbuf = np.zeros([size, n_per_node, 2], dtype = 'i')

        for idx, (A_idx, B_idx) in enumerate(product(xrange(n_split), xrange(n_split))):
            sendbuf[idx/size, idx%size, :] = A_idx, B_idx

    else:
        rbins = None
        lbox_dict = None
        split_randoms = None
        sendbuf = None

    rbins = comm.bcast(rbins, root = 0)
    lbox_dict = comm.bcast(lbox_dict, root = 0)
    Lbox = lbox_dict['Lbox']
    split_randoms = comm.bcast(split_randoms, root = 0)
    compute_idxs = comm.scatter(sendbuf, root = 0)

    RR = np.zeros((n_sub**3+1, rbins.shape[0]-1))
    for A_idx, B_idx in compute_idxs:
        print rank. A_idx, B_idx
        RR+=compute_AB(split_randoms[A_idx], split_randoms[B_idx], rbins, Lbox, n_cores = n_cores)

    all_RR = None
    if rank == 0:
        all_RR = np.empty([size, n_sub**3+1, rbins.shape[0]-1], dtype = RR.dtype)

    comm.Gather(RR, all_RR, root = 0)

    np.savetxt('RR_mpi.npy', all_RR.sum(axis = 0))


def compute_AB(randA, randB, rbins, Lbox,  n_sub = 5, n_cores = 16):

    j_index_A, N_sub_vol = cuboid_subvolume_labels(randA, n_sub, Lbox)
    j_index_B, N_sub_vol = cuboid_subvolume_labels(randB, n_sub, Lbox)

    AB = npairs_jackknife_3d(randA, randB, rbins, period=Lbox,
                    jtags1=j_index_A, jtags2=j_index_B,
                                                N_samples=N_sub_vol, num_threads=n_cores)
    AB = np.diff(AB, axis=1)

    return AB 


if __name__ == '__main__':
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    run(comm)

import numpy as np
import h5py
import pyccl as ccl
from sys import argv
from multiprocessing import Pool, cpu_count

def compute_s8(sample):
    sample = sample.astype(float)
    h = sample[5]/100
    Omb = sample[0]/h**2
    Omc = sample[1]/h**2
    w0 = sample[2]
    n_s = sample[3]
    A_s = np.exp(sample[4])*(10**(-10))
    Neff = sample[6]
    #print cosmo_param_names
    #print Omb, Omc, h, A_s, n_s, Neff, w0
    try:    
        cosmo = ccl.Cosmology(Omega_c = Omc, Omega_b = Omb, h=h, A_s = A_s, n_s = n_s, Neff = Neff, w0=w0)
        sigma_8 = ccl.sigma8(cosmo)
    except:
        return 0.,0.,0.
    
    Om = (Omb+Omc)
    s8 = sigma_8*np.sqrt(Om/0.3)
        
    return np.array([Om, sigma_8, s8])

fname = argv[1]
with h5py.File(fname, 'a') as f:
    nsteps, nwalkers = f.attrs['nsteps'], f.attrs['nwalkers'] 
    #cosmo_chain = f['chain'][:, :7]
    # TODO write in a resume option since this op is slow
    assert 's8_chain' not in f.keys(), "Already has an s8 chain, are you sure you want to overwrite?"
    f.create_dataset('s8_chain',shape=(0,3),  maxshape=(None, 3))#, dtype = np.float64)

max_chain_size = nsteps*nwalkers

from time import time
p = Pool(cpu_count())
t0 = time()

batch_size = 5000

for i in xrange(max_chain_size/batch_size+1):
    with h5py.File(fname, 'r') as f:
        cosmo_chain = f['chain'][i*batch_size:(i+1)*batch_size, :7]

    s8_chain = p.map(compute_s8, cosmo_chain)

    #print np.array(s8_chain).shape
    print time()-t0, 's', i

    with h5py.File(fname, 'a') as f:
        dset = f['s8_chain']
        dset.resize(np.min(max_chain_size, (i+1)*batch_size)), axis = 0)
        dset[-batch_size:] = s8_chain



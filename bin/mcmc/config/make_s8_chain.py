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

with h5py.File(argv[1], 'r') as f:
    cosmo_chain = f['chain'][:, :7]

from time import time
p = Pool(cpu_count())
t0 = time()
#print cosmo_chain.shape
s8_chain = p.map(compute_s8, cosmo_chain)#[:10000])

#print np.array(s8_chain).shape
print time()-t0, 's'

with h5py.File(argv[1], 'a') as f:
    if 's8_chain' in f.keys():
        del f['s8_chain']
    f.create_dataset('s8_chain', data=s8_chain)


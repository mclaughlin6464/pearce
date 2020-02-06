# I have a universe machine catalog, gonna compute statistics on them now & make sure everything seems sensible. 
# Have to do the same for shams soon
import numpy as np
from pearce.mocks.kittens import MDPL2
from sys import argv
from halotools.mock_observables import wp, delta_sigma

galcat_fname = argv[1]
basename = argv[2]
cat = MDPL2()
cat.load(1.0, particles = True)

galcat = np.load(galcat_fname)
try:
    pos = galcat['pos'][:, :3]%cat.Lbox
except ValueError: # assemble it ourselves
    #pos = np.r_[[ galcat['halo_%s'%c]%cat.Lbox for c in ['x','y','z']]].T
    pos = np.r_[[ galcat['%s'%c]%cat.Lbox for c in ['x','y','z']]].T

rbins = np.logspace(-1., 1.6, 19)
rpoints = (rbins[1:] + rbins[:-1])/2.0

# Note: make sure the luminosity cut i did makes snese with the min ptcl count I do on the emus
pi_max = 40.0
mock_wp = wp(pos / cat.h, rbins, pi_max / cat.h,\
                        period=cat.Lbox/cat.h, num_threads=8)
np.save(basename+'mock_wp.npy', mock_wp)

print 'A'
pos_m = (np.vstack([cat.halocat.ptcl_table[c] for c in ['x', 'y', 'z']]).T)%cat.Lbox

print pos.shape, pos_m.shape
import sys
sys.stdout.flush()


print 'B'
#mock_ds = calc_ds(cat, rbins, n_cores = 1, randoms = randoms) 
mock_ds = delta_sigma(pos/cat.h, pos_m/cat.h, cat.pmass/cat.h,\
                     downsampling_factor=1./cat._downsample_factor, rp_bins=rbins,
                     period=cat.Lbox/cat.h, num_threads=1, cosmology=cat.cosmology)[1] / ((1e12))#*cat.h**2)

print 'C'
np.save(basename+'mock_ds.npy', mock_ds)

import numpy as np

all_particles = np.loadtxt('/u/ki/swmclau2/des/all_particles.npy')
print all_particles.shape
print all_particles.mean(axis=0)

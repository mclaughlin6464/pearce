#!/bin/bash
from pearce.emulator.emu import OriginalRecipe, ExtraCrispy 
from pearce.emulator.trainingData import parameter, PARAMS
import numpy as np
from os import path

from guppy import hpy
training_dir = '/u/ki/swmclau2/des/PearceLHC/'

hp = hpy()

emu = ExtraCrispy(training_dir)
#or_params = PARAMS[:]
#or_params.append(parameter('r', 0,1))
#emu = OriginalRecipe(training_dir, or_params)

save_dir = '/u/ki/swmclau2/des/EmulatorMCMC/'
true_rpoints = np.log10(np.loadtxt(path.join(save_dir, 'rpoints.npy')))
y = np.loadtxt(path.join(save_dir, 'xi.npy'))
cov = np.loadtxt(path.join(save_dir, 'cov.npy'))
true_cov = cov/(np.outer(y,y)*np.log(10)**2)
#T = np.diag(np.diag(T))
true_y = np.log10(y)

print hp.heap()
chain = emu.run_mcmc(true_y, true_cov, true_rpoints, nwalkers = 20, nsteps= 2,nburn = 0,n_cores=1)

print chain.mean(axis=1)

np.savetxt(path.join(save_dir, 'chain.npy'), chain)

#!/bin/bash
#SBATCH --job-name=cosmo_mcmc
#SBATCH --time=12:00:00
#SBATCH -p iric 
#SBATCH -o /home/users/swmclau2/Git/pearce/bin/optimization/cosmo_mcmc.out
#SBATCH --ntasks=16
#SBATCH --exclusive

module load openmpi/2.1.1
python /home/users/swmclau2/Git/pearce/bin/optimization/cosmo_hyperparams_mcmc.py

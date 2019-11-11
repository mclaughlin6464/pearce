#!/bin/bash
#SBATCH --job-name=cosmo_bo
#SBATCH --time=48:00:00
#SBATCH -p iric, hns
#SBATCH -oo /home/users/swmclau2/Git/pearce/bin/optimization/cosmo_bayesian_optimizer.out
#SBATCH --ntasks=16
#SBATCH --exclusive

module load python/2.7.13
module load py-scipystack
module load hdf5/1.10.0p1
python /home/users/swmclau2/Git/pearce/bin/optimization/cosmo_bayesian_optimizer.py

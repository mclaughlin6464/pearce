#!/bin/bash
#SBATCH --job-name=pearce_mcmc_xigg_jk
#SBATCH --time=01:00:00
#SBATCH -p iric
#SBATCH -o /home/users/swmclau2/Git/pearce/bin/mcmc/pearce_mcmc_xigg_jk.out
#SBATCH --ntasks=16
#SBATCH --exclusive

module load python/2.7.13
module load py-scipystack
module load hdf5/1.10.0p1
python /home/users/swmclau2/Git/pearce/bin/mcmc/pearce_mcmc_xigg_jk.py

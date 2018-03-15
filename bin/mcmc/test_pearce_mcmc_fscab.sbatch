#!/bin/bash
#SBATCH --job-name=mcmc_vpeak_fscab
#SBATCH --time=48:00:00
#SBATCH -p normal 
#SBATCH -o mcmc_vpeak_fscab_%J.out
#SBATCH --ntasks=16
#SBATCH --mem-per-cpu=4000

python /home/swmclau2/Git/pearce/bin/mcmc/test_pearce_mcmc_fscab.py

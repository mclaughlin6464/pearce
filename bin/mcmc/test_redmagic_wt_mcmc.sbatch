#!/bin/bash
#SBATCH --job-name=mcmc_redmagic_wt
#SBATCH --time=24:00:00
#SBATCH -p iric 
#SBATCH -o mcmc_redmagic_wt_%J.out
#SBATCH --ntasks=16
#SBATCH --exclusive

python /home/swmclau2/Git/pearce/bin/mcmc/test_redmagic_wt_mcmc.py

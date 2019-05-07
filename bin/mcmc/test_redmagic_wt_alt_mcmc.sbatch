#!/bin/bash
#SBATCH --job-name=mcmc_redmagic_wt_alt
#SBATCH --time=72:00:00
#SBATCH -p iric 
#SBATCH -o mcmc_redmagic_wt_alt_%J.out
#SBATCH --ntasks=16
#SBATCH --mem-per-cpu=4000
#SBATCH --exclusive

python ~swmclau2/Git/pearce/bin/mcmc/test_redmagic_wt_alt_mcmc.py

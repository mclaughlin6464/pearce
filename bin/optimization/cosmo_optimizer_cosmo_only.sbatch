#!/bin/bash
#SBATCH --job-name=cosmo_optimizer_cosmo_only
#SBATCH --time=12:00:00
#SBATCH -p iric 
#SBATCH -o /home/users/swmclau2/Git/pearce/bin/optimization/cosmo_optimizer_cosmo_only.out
#SBATCH --ntasks=16
#SBATCH --mem-per-cpu=4000

python /home/users/swmclau2/Git/pearce/bin/optimization/cosmo_optimizer_cosmo_only.py

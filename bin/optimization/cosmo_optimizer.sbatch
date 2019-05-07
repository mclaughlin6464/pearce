#!/bin/bash
#SBATCH --job-name=cosmo_optimizer
#SBATCH --time=12:00:00
#SBATCH -p iric 
#SBATCH -o /home/users/swmclau2/Git/pearce/bin/optimization/cosmo_optimizer.out
#SBATCH --ntasks=16
#SBATCH --mem-per-cpu=4000
#SBATCH --exclusive

python /home/users/swmclau2/Git/pearce/bin/optimization/cosmo_optimizer.py

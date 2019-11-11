#!/bin/bash
#SBATCH --job-name=cosmo_grid_cosmo_only_linear
#SBATCH --time=12:00:00
#SBATCH -p iric 
#SBATCH -o /home/users/swmclau2/Git/pearce/bin/optimization/cosmo_grid_cosmo_only_linear.out
#SBATCH --ntasks=16
#SBATCH --mem-per-cpu=4000

python /home/users/swmclau2/Git/pearce/bin/optimization/cosmo_grid_cosmo_only_linear_model.py

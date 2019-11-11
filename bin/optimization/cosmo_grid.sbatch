#!/bin/bash
#SBATCH --job-name=cosmo_grid
#SBATCH --time=12:00:00
#SBATCH -p iric 
#SBATCH -o /home/users/swmclau2/Git/pearce/bin/optimization/cosmo_grid.out
#SBATCH --ntasks=16
#SBATCH --exclusive

python /home/users/swmclau2/Git/pearce/bin/optimization/cosmo_grid.py

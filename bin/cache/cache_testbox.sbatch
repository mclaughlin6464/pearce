#!/bin/bash
#SBATCH --job-name=cache_testbox
#SBATCH --time=24:00:00
#SBATCH -p iric 
#SBATCH -o /home/users/swmclau2/Git/pearce/bin/cache/cache_testbox.out
#SBATCH --ntasks=16
###SBATCH --mem-per-cpu=4000
#SBATCH --exclusive

python /home/users/swmclau2/Git/pearce/bin/cache/cache_testbox.py

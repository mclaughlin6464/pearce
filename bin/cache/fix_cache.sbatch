#!/bin/bash
#SBATCH --job-name=fix_cache
#SBATCH --time=2:00:00
#SBATCH -p iric 
#SBATCH -o /home/users/swmclau2/Git/pearce/bin/cache/fix_cache.out
#SBATCH --ntasks=16
#SBATCH --exclusive

python /home/users/swmclau2/Git/pearce/bin/cache/fix_cache.py

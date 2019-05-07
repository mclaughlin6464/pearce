#!/bin/bash
#SBATCH --job-name=cache_ds14b
#SBATCH --time=02:00:00
#SBATCH -p iric 
#SBATCH -o /home/users/swmclau2/Git/pearce/bin/cache/cache_ds14b.out
#SBATCH --exclusive

module load python/2.7.13
module load py-scipystack
module load hdf5/1.10.0p1

python /home/users/swmclau2/Git/pearce/bin/cache/cache_ds14b.py

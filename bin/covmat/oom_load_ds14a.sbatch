#!/bin/bash
#SBATCH --job-name=oom_load_ds14a
#SBATCH -p iric 
#SBATCH --time=01:00:00
#SBATCH -o oom_load_ds14a.out
#SBATCH --exclusive

module load python/2.7.13
module load py-scipystack
module load hdf5/1.10.0p1

python /home/users/swmclau2/Git/pearce/bin/covmat/oom_load_ds14a.py

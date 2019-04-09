#!/bin/bash
#SBATCH --job-name=test_load_ds14a
#SBATCH -p iric 
#SBATCH --time=24:00:00
#SBATCH -o test_load_ds14a.out
#SBATCH --exclusive

module load python/2.7.13
module load py-scipystack
module load hdf5/1.10.0p1

python /home/users/swmclau2/Git/pearce/bin/covmat/test_load_ds14a.py

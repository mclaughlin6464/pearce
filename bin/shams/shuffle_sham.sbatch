#!/bin/bash
#SBATCH --job-name=shuffle_sham
#SBATCH --time=02:00:00
#SBATCH -p iric,kipac,owners
#SBATCH -o shuffle_sham.out
#SBATCH -c 8 
#SBATCH --mem=MaxMemPerCPU
##SBATCH --exclusive

module load python/2.7.13
module load py-scipystack
module load hdf5/1.10.0p1

python /home/users/swmclau2/Git/pearce/bin/shams/shuffle_sham.py

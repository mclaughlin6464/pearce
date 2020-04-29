#!/bin/bash
#SBATCH --job-name=make_sham
#SBATCH --time=02:00:00
#SBATCH -p iric,kipac 
#SBATCH -o make_sham.out
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=MaxMemPerCPU

###SBATCH --exclusive

module load python/2.7.13
module load py-scipystack
module load hdf5/1.10.0p1

python /home/users/swmclau2/Git/pearce/bin/shams/make_sham.py

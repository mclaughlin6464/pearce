#!/bin/bash
#SBATCH --job-name=compute_emu_covmat
#SBATCH -p iric 
#SBATCH --time=08:00:00
#SBATCH -o compute_emu_covmat.out 
####SBATCH --exclusive
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=MaxMemPerCPU

module load python/2.7.13
module load py-scipystack
module load hdf5/1.10.0p1

python /home/users/swmclau2/Git/pearce/bin/covmat/compute_emu_covmat.py

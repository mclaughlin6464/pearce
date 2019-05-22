#!/bin/bash
#SBATCH --job-name=test_emu_acc
#SBATCH -p iric 
#SBATCH --time=04:00:00
#SBATCH -o test_emu_acc.out 
####SBATCH --exclusive
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=MaxMemPerCPU

module load python/2.7.13
module load py-scipystack
module load hdf5/1.10.0p1

python /home/users/swmclau2/Git/pearce/bin/covmat/test_emu_acc.py

#!/bin/bash
#SBATCH --job-name=compute_rr_pairs_gg
#SBATCH -p iric 
#SBATCH --time=04:00:00
#SBATCH -o compute_rr_pairs_gg.out 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G

module load python/2.7.13
module load py-scipystack
module load hdf5/1.10.0p1

python /home/users/swmclau2/Git/pearce/bin/covmat/compute_rr_pairs_gg.py

#!/bin/bash
#SBATCH --job-name=cache_halocat_no_downsample
#SBATCH --time=48:00:00
#SBATCH -p iric 
#SBATCH -o /home/users/swmclau2/Git/pearce/bin/cache/cache_halocat_no_downsample.out
#SBATCH --ntasks=16
#SBATCH --exclusive

python /home/users/swmclau2/Git/pearce/bin/cache/cache_halocat_no_downsample.py

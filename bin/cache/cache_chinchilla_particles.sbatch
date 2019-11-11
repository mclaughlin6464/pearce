#!/bin/bash
#SBATCH --job-name=cache_halocat
#SBATCH --time=4:00:00
#SBATCH -p iric 
#SBATCH -o /home/users/swmclau2/Git/pearce/bin/cache/cache_chinchilla.out
#SBATCH --ntasks=16
#SBATCH --mem-per-cpu=4000

python /home/users/swmclau2/Git/pearce/bin/cache/cache_chinchilla_particles.py

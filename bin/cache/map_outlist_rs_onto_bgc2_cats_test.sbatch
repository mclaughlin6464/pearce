#!/bin/bash
#SBATCH --job-name=map_outlist_rs_onto_bgc2_cats_test
#SBATCH --time=24:00:00
#SBATCH -p iric 
#SBATCH -o /home/users/swmclau2/Git/pearce/bin/cache/map_outlist_rs_onto_bgc2_cats_test.out
#SBATCH --ntasks=16
#SBATCH --exclusive

python /home/users/swmclau2/Git/pearce/bin/cache/map_outlist_rs_onto_bgc2_cats_test.py

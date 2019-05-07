#!/bin/bash
#SBATCH --job-name=write_lsd_boxes
#SBATCH --time=1:00:00
#SBATCH -p normal 
#SBATCH -o /home/users/swmclau2/Git/pearce/bin/cache/write_lsd_boxes.out
#SBATCH --nodes=1
#SBATCH --exclusive

python /home/users/swmclau2/Git/pearce/bin/cache/write_lsd_boxes.py

#!/bin/bash
#SBATCH --job-name=wt_cosmo_trainer
#SBATCH --time=01:00:00
#SBATCH -p iric, owners 
#SBATCH -o /home/users/swmclau2/Git/pearce/bin/trainer/wt_cosmo_trainer_%J.out
#SBATCH --exclusive

#module load openmpi/2.1.1
srun python ~swmclau2/Git/pearce/pearce/emulator/trainer.py ~swmclau2/Git/pearce/bin/trainer/wt_cosmo_trainer.yaml --rerun

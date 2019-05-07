#!/bin/bash
#SBATCH --job-name=xi_gm_trainer
#SBATCH --time=00:30:00
#SBATCH -p iric 
#SBATCH -o /home/users/swmclau2/Git/pearce/bin/trainer/xi_gm_trainer_%J.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --exclusive

srun python ~swmclau2/Git/pearce/pearce/emulator/trainer.py ~swmclau2/Git/pearce/bin/trainer/xi_gm_cosmo_trainer.yaml --rerun

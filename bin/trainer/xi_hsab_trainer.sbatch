#!/bin/bash
#SBATCH --job-name=xi_hsab_trainer
#SBATCH --time=00:30:00
#SBATCH -p iric 
#SBATCH -o /home/users/swmclau2/Git/pearce/bin/trainer/xi_hsab_trainer_%J.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4000

srun python ~swmclau2/Git/pearce/pearce/emulator/trainer.py ~swmclau2/Git/pearce/bin/trainer/xi_cosmo_trainer_hsab.yaml --rerun

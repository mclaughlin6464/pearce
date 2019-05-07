#!/bin/bash
#SBATCH --job-name=xi_test_trainer
#SBATCH --time=01:00:00
#SBATCH -p iric 
#SBATCH -o /home/users/swmclau2/Git/pearce/bin/trainer/xi_cosmo_trainer_test_%J.out
#SBATCH --exclusive

srun python ~swmclau2/Git/pearce/pearce/emulator/trainer.py ~swmclau2/Git/pearce/bin/trainer/xi_cosmo_trainer_test.yaml --rerun 

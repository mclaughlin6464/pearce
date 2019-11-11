#!/bin/bash
#SBATCH --job-name=xi_test_trainer_lowmsat
#SBATCH --time=00:10:00
#SBATCH -p iric 
#SBATCH -o /home/users/swmclau2/Git/pearce/bin/trainer/xi_cosmo_trainer_test_lowmsat_%J.out
#SBATCH --exclusive

srun python ~swmclau2/Git/pearce/pearce/emulator/trainer.py ~swmclau2/Git/pearce/bin/trainer/xi_cosmo_trainer_test.yaml 

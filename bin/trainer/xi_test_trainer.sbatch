#!/bin/bash
#SBATCH --job-name=xi_test_trainer
#SBATCH --time=02:00:00
#SBATCH -p iric 
#SBATCH -o /home/users/swmclau2/Git/pearce/bin/trainer/xi_test_trainer_%J.out
#SBATCH --exclusive

srun python ~swmclau2/Git/pearce/pearce/emulator/trainer.py ~swmclau2/Git/pearce/bin/trainer/xi_test_trainer.yaml

#!/bin/bash
#SBATCH --job-name=xi_gm_test_trainer
#SBATCH --time=00:30:00
#SBATCH -p iric 
#SBATCH -o /home/users/swmclau2/Git/pearce/bin/trainer/xi_gm_test_trainer_%J.out
#SBATCH --exclusive

python ~swmclau2/Git/pearce/pearce/emulator/trainer.py ~swmclau2/Git/pearce/bin/trainer/xi_gm_cosmo_test_trainer.yaml  --rerun

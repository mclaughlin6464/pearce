#!/bin/bash
#SBATCH --job-name=test_trainer
#SBATCH --time=01:00:00
#SBATCH -p iric 
#SBATCH -o /home/users/swmclau2/Git/pearce/bin/trainer/test_trainer2_%J.out
#SBATCH --nodes=40
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4000
#SBATCH --exclusive

srun python ~swmclau2/Git/pearce/pearce/emulator/trainer.py test_trainer.yaml 

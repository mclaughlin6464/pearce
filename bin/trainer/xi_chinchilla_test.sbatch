#!/bin/bash
#SBATCH --job-name=xi_chinchilla_test
#SBATCH --time=24:00:00
#SBATCH -p iric 
#SBATCH -o /home/users/swmclau2/Git/pearce/bin/trainer/xi_chinchilla_test_%J.out
#SBATCH --nodes=5
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4000
#SBATCH --exclusive

srun python ~swmclau2/Git/pearce/pearce/emulator/trainer.py ~swmclau2/Git/pearce/bin/trainer/xi_chinchilla_test.yaml

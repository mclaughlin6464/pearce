#!/bin/bash
#SBATCH --job-name=xi_cosmo_trainer
#SBATCH --time=24:00:00
#SBATCH -p iric 
#SBATCH -o /home/users/swmclau2/Git/pearce/bin/trainer/xi_cosmo_trainer_%J.out
#SBATCH --nodes=40
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive

#module load openmpi/2.1.1
srun python ~swmclau2/Git/pearce/pearce/emulator/trainer.py ~swmclau2/Git/pearce/bin/trainer/xi_cosmo_trainer.yaml

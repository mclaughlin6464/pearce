#!/bin/bash
#SBATCH --job-name=xi_cosmo_trainer2
#SBATCH --time=00:20:00
#SBATCH -p iric 
#SBATCH -o /home/users/swmclau2/Git/pearce/bin/trainer/xi_cosmo_trainer2_%J.out
###SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --tasks-per-node=2
###SBATCH --cpus-per-task=4
###SBATCH --mem-per-cpu=MaxMemPerCPU
#SBATCH --exclusive

#module load openmpi/2.1.1
srun python ~swmclau2/Git/pearce/pearce/emulator/trainer.py ~swmclau2/Git/pearce/bin/trainer/xi_cosmo_trainer2.yaml

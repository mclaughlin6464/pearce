#!/bin/bash
#SBATCH --job-name=pmesh_test
#SBATCH --time=01:00:00
#SBATCH -p regular
#SBATCH -A cosmosim
#SBATCH -N 1


module load openmpi
srun python ~swmclau2/Git/pearce/pearce/sims/submit_sims.py pmesh_test.yaml 

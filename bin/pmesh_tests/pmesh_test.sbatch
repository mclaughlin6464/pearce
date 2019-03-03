#!/bin/bash
#SBATCH --job-name=pmesh_test
#SBATCH --time=01:00:00
#SBATCH -p iric, owners 
#SBATCH -o pmest_test.out

module load openmpi/2.1.1
srun python ~swmclau2/Git/pearce/pearce/sims/submit_sims.py pmesh_test.yaml 

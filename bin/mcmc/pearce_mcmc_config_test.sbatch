#!/bin/bash
#SBATCH --job-name=pearce_mcmc_config_test
#SBATCH --time=01:00:00
#SBATCH -p iric
#SBATCH -o /home/users/swmclau2/Git/pearce/bin/mcmc/pearce_mcmc_config_test.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=MaxMemPerCPU

module load python/2.7.13
module load py-scipystack
module load hdf5/1.10.0p1
python /home/users/swmclau2/Git/pearce/pearce/inference/initialize_mcmc.py example_mcmc_config.yaml
#python /home/users/swmclau2/Git/pearce/pearce/inference/run_mcmc.py /scratch/users/swmclau2/PearceMCMC/pearce_mcmc_joint_ind.hdf5

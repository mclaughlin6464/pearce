#!/bin/bash
#SBATCH --job-name=nh_gg_mcmc_config
#SBATCH --time=24:00:00
#SBATCH -p iric
#SBATCH -o /home/users/swmclau2/Git/pearce/bin/mcmc/nh_gg_mcmc_config.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=MaxMemPerCPU

module load python/2.7.13
module load py-scipystack
module load hdf5/1.10.0p1
python /home/users/swmclau2/Git/pearce/bin/covmat/compute_emu_covmat.py
python /home/users/swmclau2/Git/pearce/pearce/inference/initialize_mcmc.py nh_gg_mcmc_config.yaml
python /home/users/swmclau2/Git/pearce/pearce/inference/run_mcmc.py nh_gg_mcmc_config.yaml


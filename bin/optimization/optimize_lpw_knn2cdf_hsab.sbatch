#!/bin/bash
#SBATCH --job-name=optimize_lpw_wp_hsab
#SBATCH -p kipac,iric,hns,owners
#SBATCH --time=24:00:00
#SBATCH -o optimize_lpw_knn2cdf_hsab.out 
####SBATCH --exclusive
#SBATCH --cpus-per-task=8
####SBATCH --mem-per-cpu=MaxMemPerCPU

module load python/2.7.13
module load py-scipystack
module load hdf5/1.10.0p1

python /home/users/swmclau2/Git/pearce/bin/optimization/optimize_lpw.py                       ~/scratch/Aemulators/knn_2_cdf_hsab/PearceKNN2CDFCutHSABCosmo.hdf5                       ~/scratch/Aemulators/knn_2_cdf_hsab_test/PearceKNN2CDFCutHSABCosmoTest.hdf5                       knn_2_cdf_cut_hsab_emu_cov_lpw.npy                       2 3 1

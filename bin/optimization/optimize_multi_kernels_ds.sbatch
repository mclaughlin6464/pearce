#!/bin/bash
#SBATCH --job-name=optimize_nh
#SBATCH -p iric 
#SBATCH --time=8:00:00
#SBATCH -o optimize_multi_kernels_ds.out 
####SBATCH --exclusive
#SBATCH --cpus-per-task=8
####SBATCH --mem-per-cpu=MaxMemPerCPU

module load python/2.7.13
module load py-scipystack
module load hdf5/1.10.0p1

#python /home/users/swmclau2/Git/pearce/bin/optimization/optimize_nh.py /home/users/swmclau2/scratch/xi_gg_zheng07/PearceXiggCosmoCorrAB.hdf5
#python /home/users/swmclau2/Git/pearce/bin/optimization/optimize_nh.py ~/scratch/cic_zheng07_v2/PearceCICCosmo.hdf5 /scratch/users/swmclau2/cic_zheng07_test_v2/PearceCICCosmoTest.hdf5
#python /home/users/swmclau2/Git/pearce/bin/optimization/optimize_multi_kernels.py ~/scratch/wp_zheng07/PearceWpCosmo.hdf5 ~/scratch/wp_zheng07_test/PearceWpCosmoTest.hdf5
python /home/users/swmclau2/Git/pearce/bin/optimization/optimize_multi_kernels.py ~/scratch/ds_zheng07/PearceDsCosmo.hdf5 ~/scratch/ds_zheng07_test/PearceDsCosmoTest.hdf5 ds_hod_emu_cov.npy 4 0

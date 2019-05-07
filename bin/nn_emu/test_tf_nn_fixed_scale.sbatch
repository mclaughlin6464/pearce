#!/bin/bash
#SBATCH --job-name=test_nn_emu_fs
#SBATCH --time=24:00:00
#SBATCH --gres gpu:1
#SBATCH -p iric 
#SBATCH -o test_tf_nn_fixed_scale.out 

module load py-tensorflow
python test_tf_nn_fixed_scale.py 

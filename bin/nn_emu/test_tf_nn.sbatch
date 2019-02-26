#!/bin/bash
#SBATCH --job-name=test_nn_emu
#SBATCH --time=04:00:00
#SBATCH --gres gpu:2
#SBATCH --ntasks 8
#SBATCH -p gpu 
#SBATCH -o test_nn_emu.out 

module load py-tensorflow
python test_tf_nn.py 

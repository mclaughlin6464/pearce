#!/bin/bash
#SBATCH --job-name=test_nn_emu
#SBATCH --time=24:00:00
#SBATCH --gres gpu:1
#SBATCH -p iric 
#SBATCH -o test_nn_emu.out 

module load py-tensorflow
python test_tf_nn.py 

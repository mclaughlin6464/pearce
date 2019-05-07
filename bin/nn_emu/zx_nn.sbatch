#!/bin/bash
#SBATCH --job-name=zx_nn
#SBATCH --time=24:00:00
#SBATCH --gres gpu:1
#SBATCH -p iric
#SBATCH -o zx_nn.out 

module load py-tensorflow
python zx_nn.py 

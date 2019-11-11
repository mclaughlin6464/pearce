#!/bin/bash
#SBATCH --job-name=test_nn_emu_novel
#SBATCH --time=03:00:00
#SBATCH --gres gpu:2
#SBATCH --ntasks 8
#SBATCH -p gpu 
#SBATCH -o test_tf_nn_novel.out 

module load py-tensorflow
python test_tf_nn_novel.py 

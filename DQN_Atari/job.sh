#!/bin/bash

#SBATCH --mail-user=svshirke@wpi.edu
#SBATCH --mail-type=ALL

#SBATCH -J break_out_NN
#SBATCH --output=/home/svshirke/RLdqn/Project3/logs/break_out_NN%j.out
#SBATCH --error=/home/svshirke/RLdqn/Project3/logs/break_out_NN%j.err

#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -C A100|V100

#SBATCH -p short
#SBATCH -t 23:59:00





echo "Starting myscript"
source activate pt
python main.py --train_dqn
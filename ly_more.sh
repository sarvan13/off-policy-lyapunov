#!/bin/bash
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=20:00:0
#SBATCH --mail-user=sarvan13@uvic.ca
#SBATCH --mail-type=ALL
#SBATCH --account=def-danielac
#SBATCH --gpus-per-node=1

cd ~/projects/def-danielac/sarvan13/off-policy-lyapunov
module purge
module load python/3.10.13
module load mujoco
source ~/RLENV/bin/activate

python train_ly.py --N 2048 --n_steps 10000000 --batch_size 64


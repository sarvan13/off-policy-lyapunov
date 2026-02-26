#!/bin/bash
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=10:00:0
#SBATCH --mail-user=sarvan13@uvic.ca
#SBATCH --mail-type=ALL
#SBATCH --account=def-danielac
#SBATCH --gpus-per-node=1
#SBATCH --array=0-39

cd ~/projects/def-danielac/sarvan13/thesis-lyapunov/off-policy-lyapunov
module purge
module load python/3.10.13
module load mujoco
source ~/MujocoENV/bin/activate

MU_VALUES=(0.01 0.1 0.5 0.75)

MU_INDEX=$((SLURM_ARRAY_TASK_ID / 10))
MU=${MU_VALUES[$MU_INDEX]}
SEED=$(((SLURM_ARRAY_TASK_ID % 10) + 1))

python clean_lppo.py --env-id Quadrotor-Still-v1 --num-envs 8 --ent-coef 0.001 --save_model --total-timesteps 15000000 --seed $SEED --mu $MU


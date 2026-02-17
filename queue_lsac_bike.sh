#!/bin/bash
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=23:00:0
#SBATCH --mail-user=sarvan13@uvic.ca
#SBATCH --mail-type=ALL
#SBATCH --account=def-danielac
#SBATCH --gpus-per-node=1
#SBATCH --array=11-20

cd ~/projects/def-danielac/sarvan13/off-policy-lyapunov
module purge
module load python/3.10.13
module load mujoco
source ~/MujocoENV/bin/activate

python run_off_policy.py --modelType lsac --env Bicycle-v1 --n_steps 3000000 --seed $SLURM_ARRAY_TASK_ID


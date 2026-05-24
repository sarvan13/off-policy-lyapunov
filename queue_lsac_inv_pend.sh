#!/bin/bash
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=10:00:0
#SBATCH --mail-user=sarvan13@uvic.ca
#SBATCH --mail-type=ALL
#SBATCH --account=def-danielac
#SBATCH --gpus-per-node=1
# --- NEW ARRAY RANGE: 40 tasks (0 to 39) ---
#SBATCH --array=0-9

cd ~/projects/def-danielac/sarvan13/thesis-reg/off-policy-lyapunov
module purge
module load python/3.10.13
module load mujoco
source ~/MujocoENV/bin/activate

# 1. Define your mu values in a bash array

# 2. Logic to pick the index (0, 1, 2, or 3) and the seed
# % is modulo, / is integer division
SEED=$(( (SLURM_ARRAY_TASK_ID % 10) + 11 ))

# 3. Pass the new mu parameter to your script
python -u run_off_policy.py \
    --modelType lsac \
    --env InvertedDoublePendulum-v5 \
    --n_steps 1000000 \
    --seed $SEED \
    --mu 0.01

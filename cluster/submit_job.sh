#!/bin/bash
#SBATCH --job-name=snake_4_a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err

# 1. Create logs directory if it doesn't exist
mkdir -p logs

# 2. Load modules and activate environment
module load Miniforge3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate snake_env

# 3. Setup training command
# We use torchrun for easy DDP orchestration
# --nproc_per_node should match the number of GPUs requested (--gres)
CMD="python3 -m torch.distributed.run \
    --nproc_per_node=4 \
    my_nn_snake/core/train.py \
    --total-steps 500000000 \
    --load best \
    --log-interval 1 \
    --save-interval 50 \
    --verbose"

echo "🚀 Launching Battlesnake Training on 4x A100..."
echo "Running: $CMD"

$CMD

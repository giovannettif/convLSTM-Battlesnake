#!/bin/bash
#SBATCH --job-name=snake_20_a100
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err

# 1. Create logs directory
mkdir -p logs

# 2. Load environment
module load Miniforge3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate snake_env

# 3. Multi-Node DDP Setup
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12345

echo "🌐 Master Node: $MASTER_ADDR"
echo "🚀 Launching Battlesnake Training on 20x A100 (5 Nodes)..."

# 4. Launch with torchrun
python3 -m torch.distributed.run \
    --nnodes=5 \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    my_nn_snake/core/train.py \
    --total-steps 500000000 \
    --load best \
    --log-interval 1 \
    --save-interval 50 \
    --verbose

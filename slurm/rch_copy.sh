#!/bin/bash
#SBATCH --job-name=rch-copy
#SBATCH --partition=serial
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/hpc/11/xiar3/MetricTrust/runs/rch-copy-%j.out
# Section 16B: copy the new models' final checkpoints to storage before evaluation.
bash /mmfs1/storage/users/xiar3/exp/residual-calibrated-hedging/latency/copy_new_ckpts.sh
ls -R /mmfs1/storage/users/xiar3/exp/residual-calibrated-hedging/checkpoints_new | grep -c "\.pt$"

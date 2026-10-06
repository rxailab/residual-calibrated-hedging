#!/bin/bash
#SBATCH --job-name=mt-lat-diff
#SBATCH --partition=astro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --nice=100
#SBATCH --output=/scratch/hpc/11/xiar3/MetricTrust/runs/lat-diff-%A_%a.out
# Side paper Study 3, diffusion part (latency/run_diffusion.py): one (task, seed batch, arm) per
# array task, 5 episodes, on astro (L40 / L40S only, so every arm shares one GPU architecture and
# sampler precision).  Single-environment latency is measured on a V100 by slurm/latency_timing.sh.
#   sbatch --array=0-23 slurm/latency_diffusion.sh
set -e
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYTHONPATH=${RCH_ROOT:-/mmfs1/storage/users/xiar3/exp/residual-calibrated-hedging}
PY=/mmfs1/storage/users/xiar3/envs/metrictrust/bin/python
R=${RCH_ROOT:-/mmfs1/storage/users/xiar3/exp/residual-calibrated-hedging}
CK=${CKPT_KEEP:-/mmfs1/storage/users/xiar3/exp/MetricTrust/checkpoints_keep}   # model checkpoints (not in this repo)
i=${SLURM_ARRAY_TASK_ID:-0}
nvidia-smi --query-gpu=name --format=csv,noheader
LINE=$(sed -n "$(( i + 1 ))p" $R/latency/${LIST:-runs_diffusion.txt})
echo "$LINE"
$PY $R/latency/run_diffusion.py $LINE

#!/bin/bash
#SBATCH --job-name=mt-lat-wall
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:tesla_v100-pcie-32gb:1
#SBATCH --exclusive
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --nice=100
#SBATCH --output=/scratch/hpc/11/xiar3/MetricTrust/runs/lat-wall-%A_%a.out
# Side paper Study 4 (latency/wallclock.py): RSSM seed0 on cartpole-swingup in real time,
# one exclusive V100 node so no other job shares the GPU or the CPUs.
# Array index 0: population 512, 6 iterations; 1: population 128, 2 iterations.
#   sbatch --array=0-1 slurm/latency_wallclock.sh
set -e
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYTHONPATH=${RCH_ROOT:-/mmfs1/storage/users/xiar3/exp/residual-calibrated-hedging}
PY=/mmfs1/storage/users/xiar3/envs/metrictrust/bin/python
R=${RCH_ROOT:-/mmfs1/storage/users/xiar3/exp/residual-calibrated-hedging}
CK=${CKPT_KEEP:-/mmfs1/storage/users/xiar3/exp/MetricTrust/checkpoints_keep}   # model checkpoints (not in this repo)
CFG=("--pop 512 --iters 6" "--pop 128 --iters 2")
nvidia-smi --query-gpu=name --format=csv,noheader
$PY $R/latency/wallclock.py $CK/dmc/rssm/cartpole-swingup/seed0/step0060000.pt \
  ${CFG[${SLURM_ARRAY_TASK_ID:-0}]} --episodes 30

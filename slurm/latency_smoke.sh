#!/bin/bash
#SBATCH --job-name=mt-lat-smoke
#SBATCH --partition=astro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/hpc/11/xiar3/MetricTrust/runs/lat-smoke-%j.out
# Smoke test of latency/run_dmc.py with real EGL rendering on the node it lands on.
set -e
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYTHONPATH=${RCH_ROOT:-/mmfs1/storage/users/xiar3/exp/residual-calibrated-hedging} DMC_ENV_WORKERS=6
PY=/mmfs1/storage/users/xiar3/envs/metrictrust/bin/python
R=${RCH_ROOT:-/mmfs1/storage/users/xiar3/exp/residual-calibrated-hedging}
CK=${CKPT_KEEP:-/mmfs1/storage/users/xiar3/exp/MetricTrust/checkpoints_keep}   # model checkpoints (not in this repo)
nvidia-smi --query-gpu=name --format=csv,noheader
$PY $R/latency/run_dmc.py $CK/dmc/rssm/cartpole-swingup/seed0/step0060000.pt \
  --arms paused1,cpp4 --episodes 50 --T 40 --out /scratch/hpc/11/xiar3/MetricTrust/runs/lat_smoke_out

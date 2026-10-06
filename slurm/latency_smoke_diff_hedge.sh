#!/bin/bash
#SBATCH --job-name=mt-lat-smoke
#SBATCH --partition=astro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/hpc/11/xiar3/MetricTrust/runs/lat-smoke-diff-%j.out
# Smoke test of latency/run_diffusion.py: tiny planner, 2 episodes of 32 decisions, all arms.
set -e
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYTHONPATH=${RCH_ROOT:-/mmfs1/storage/users/xiar3/exp/residual-calibrated-hedging}
PY=/mmfs1/storage/users/xiar3/envs/metrictrust/bin/python
R=${RCH_ROOT:-/mmfs1/storage/users/xiar3/exp/residual-calibrated-hedging}
CK=${CKPT_KEEP:-/mmfs1/storage/users/xiar3/exp/MetricTrust/checkpoints_keep}   # model checkpoints (not in this repo)
nvidia-smi --query-gpu=name --format=csv,noheader
$PY $R/latency/run_diffusion.py $CK/dmc/diffusion/dmcontrol/seed0/step0032000.pt \
  --task cup-catch --episodes 2 --T 48 --set pop=4 iters=1 --arms hcpp8k1,hcpp8k4,hcpp16k4 \
  --out /scratch/hpc/11/xiar3/MetricTrust/runs/lat_smoke_diff_out

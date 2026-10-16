#!/bin/bash
#SBATCH --job-name=mt-lat-slewm
#SBATCH --partition=astro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/hpc/11/xiar3/MetricTrust/runs/lat-slewm-%j.out
# Smoke test: latency/lewm.py on PushT (10 episodes, all arm kinds) and the real-time rch arm.
set -e
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl SDL_VIDEODRIVER=dummy STABLEWM_HOME=/scratch/hpc/11/xiar3/lewm
R=/mmfs1/storage/users/xiar3/exp/residual-calibrated-hedging
O=/scratch/hpc/11/xiar3/MetricTrust/runs/lat_smoke_lewm
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
PYTHONPATH=$R /mmfs1/storage/users/xiar3/envs/rch-lewm/bin/python $R/latency/lewm.py pusht \
  --arms paused1,cpp1,rch1k4,aligned2,cpp2,rch2k4,paused2 --episodes 10 --chunk 10 --out $O

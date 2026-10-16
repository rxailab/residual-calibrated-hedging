#!/bin/bash
#SBATCH --job-name=mt-lat-tlewm
#SBATCH --partition=astro
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/hpc/11/xiar3/MetricTrust/runs/lat-tlewm-%j.out
# Batch-1 latency of LeWM planning calls (latency/timing_lewm.py), L40S, shared node.
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl SDL_VIDEODRIVER=dummy STABLEWM_HOME=${STABLEWM_HOME:-/scratch/hpc/11/xiar3/lewm}
R=${RCH_ROOT:-/mmfs1/storage/users/xiar3/exp/residual-calibrated-hedging}
nvidia-smi --query-gpu=name --format=csv,noheader
PYTHONPATH=$R /mmfs1/storage/users/xiar3/envs/rch-lewm/bin/python $R/latency/timing_lewm.py

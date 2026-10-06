#!/bin/bash
#SBATCH --job-name=mt-lat-err
#SBATCH --partition=astro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/hpc/11/xiar3/MetricTrust/runs/lat-err-%j.out
# Method-design diagnostic (latency/error_model.py) on the nine seed0 latent models.
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYTHONPATH=${RCH_ROOT:-/mmfs1/storage/users/xiar3/exp/residual-calibrated-hedging}
PY=/mmfs1/storage/users/xiar3/envs/metrictrust/bin/python
R=${RCH_ROOT:-/mmfs1/storage/users/xiar3/exp/residual-calibrated-hedging}
CK=${CKPT_KEEP:-/mmfs1/storage/users/xiar3/exp/MetricTrust/checkpoints_keep}   # model checkpoints (not in this repo)
nvidia-smi --query-gpu=name --format=csv,noheader
while read -r CK; do $PY $R/latency/error_model.py $CK $ARGS; done < $R/latency/${LIST:-runs_timing.txt}

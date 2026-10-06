#!/bin/bash
#SBATCH --job-name=mt-lat-th
#SBATCH --partition=astro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/hpc/11/xiar3/MetricTrust/runs/lat-th-%j.out
# Batch-1 latency of hedged planning (latency/timing_hedge.py) for three models.  Indicative:
# not an exclusive node; the exclusive per-GPU timing jobs (slurm/latency_timing.sh) are the
# reference measurements.
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYTHONPATH=${RCH_ROOT:-/mmfs1/storage/users/xiar3/exp/residual-calibrated-hedging}
PY=/mmfs1/storage/users/xiar3/envs/metrictrust/bin/python
R=${RCH_ROOT:-/mmfs1/storage/users/xiar3/exp/residual-calibrated-hedging}
CK=${CKPT_KEEP:-/mmfs1/storage/users/xiar3/exp/MetricTrust/checkpoints_keep}   # model checkpoints (not in this repo)
nvidia-smi --query-gpu=name --format=csv,noheader
while read -r CK; do $PY $R/latency/timing_hedge.py $CK; done < $R/latency/runs_timing_hedge.txt

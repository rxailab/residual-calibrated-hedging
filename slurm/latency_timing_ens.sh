#!/bin/bash
#SBATCH --job-name=mt-lat-tens
#SBATCH --partition=astro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/hpc/11/xiar3/MetricTrust/runs/lat-tens-%j.out
# Batch-1 latency, memory and parameters of the ensemble baseline against RCH
# (latency/timing_ensemble.py), for the three cells of slurm/latency_timing_hedge.sh.  Indicative,
# like that job: not an exclusive node.  Pass --constraint to choose the GPU type.
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYTHONPATH=${RCH_ROOT:-/mmfs1/storage/users/xiar3/exp/residual-calibrated-hedging}
export CKPT_KEEP=${CKPT_KEEP:-/mmfs1/storage/users/xiar3/exp/MetricTrust/checkpoints_keep}   # model checkpoints (not in this repo)
PY=/mmfs1/storage/users/xiar3/envs/metrictrust/bin/python
R=${RCH_ROOT:-/mmfs1/storage/users/xiar3/exp/residual-calibrated-hedging}
nvidia-smi --query-gpu=name --format=csv,noheader
for C in "rssm walker-walk" "jepa cup-catch" "tdmpc2 cup-catch"; do $PY $R/latency/timing_ensemble.py $C; done

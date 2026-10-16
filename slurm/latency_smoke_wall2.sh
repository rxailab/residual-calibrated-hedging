#!/bin/bash
#SBATCH --job-name=mt-lat-swall
#SBATCH --partition=astro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/hpc/11/xiar3/MetricTrust/runs/lat-swall-%j.out
# Smoke test of the real-time rch arm and calibration (60 decisions, 1 episode, results to scratch).
set -e
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl
R=/mmfs1/storage/users/xiar3/exp/residual-calibrated-hedging
nvidia-smi --query-gpu=name --format=csv,noheader
PYTHONPATH=$R /mmfs1/storage/users/xiar3/envs/metrictrust/bin/python /scratch/hpc/11/xiar3/MetricTrust/runs/smoke/smoke_wall2.py

#!/bin/bash
#SBATCH --job-name=rch-ldiag
#SBATCH --partition=astro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --nice=100
#SBATCH --output=/scratch/hpc/11/xiar3/MetricTrust/runs/rch-ldiag-%j.out
# Section 16C exploratory: offline hedging value for LeWM (latency/lewm_diag.py).
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl SDL_VIDEODRIVER=dummy STABLEWM_HOME=/scratch/hpc/11/xiar3/lewm
R=/mmfs1/storage/users/xiar3/exp/residual-calibrated-hedging
PYTHONPATH=$R /mmfs1/storage/users/xiar3/envs/rch-lewm/bin/python $R/latency/lewm_diag.py

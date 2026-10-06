#!/bin/bash
#SBATCH --job-name=mt-lat-smoke
#SBATCH --partition=astro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/hpc/11/xiar3/MetricTrust/runs/lat-smoke-de-%j.out
# Smoke test of the delay-error arms (latency/delay_error.py) with real rendering.
set -e
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYTHONPATH=${RCH_ROOT:-/mmfs1/storage/users/xiar3/exp/residual-calibrated-hedging} DMC_ENV_WORKERS=2
PY=/mmfs1/storage/users/xiar3/envs/metrictrust/bin/python
R=${RCH_ROOT:-/mmfs1/storage/users/xiar3/exp/residual-calibrated-hedging}
CK=${CKPT_KEEP:-/mmfs1/storage/users/xiar3/exp/MetricTrust/checkpoints_keep}   # model checkpoints (not in this repo)
$PY $R/latency/run_dmc.py $CK/dmc/tdmpc2/walker-walk/seed0/step0100000.pt \
  --arms ecpp4k4,ecpp4k4h1.0,ecpp4k4r1.0,mcpp4,scpp4k4 --episodes 4 --T 40 --seed0 970000 \
  --out /scratch/hpc/11/xiar3/MetricTrust/runs/lat_smoke_de_out

#!/bin/bash
#SBATCH --job-name=mt-lat-sens
#SBATCH --partition=astro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/hpc/11/xiar3/MetricTrust/runs/lat-sens-%j.out
# Smoke test of the ensemble and particle arms (latency/ensemble.py, pcpp in latency/run_dmc.py):
# 40 decisions, output to scratch.
set -e
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYTHONPATH=${RCH_ROOT:-/mmfs1/storage/users/xiar3/exp/residual-calibrated-hedging} DMC_ENV_WORKERS=6
export CKPT_KEEP=${CKPT_KEEP:-/mmfs1/storage/users/xiar3/exp/MetricTrust/checkpoints_keep}
PY=/mmfs1/storage/users/xiar3/envs/metrictrust/bin/python
R=${RCH_ROOT:-/mmfs1/storage/users/xiar3/exp/residual-calibrated-hedging}
O=/scratch/hpc/11/xiar3/MetricTrust/runs/lat_smoke_ens
nvidia-smi --query-gpu=name --format=csv,noheader
for F in tdmpc2 jepa rssm; do
  T=cup-catch
  $PY $R/latency/ensemble.py $F $T --arms ens4,ens4k4 --episodes 50 --T 40 --out $O
done
$PY $R/latency/run_dmc.py $CKPT_KEEP/dmc/rssm/cup-catch/seed0/step0060000.pt --arms pcpp4k4,hcpp4k4 --episodes 50 --T 40 --out $O
$PY $R/latency/timing_ensemble.py rssm cup-catch --calls 5

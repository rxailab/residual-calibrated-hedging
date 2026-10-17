#!/bin/bash
#SBATCH --job-name=mt-lat-ens
#SBATCH --partition=astro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --nice=100
#SBATCH --output=/scratch/hpc/11/xiar3/MetricTrust/runs/lat-ens-%A_%a.out
# Ensemble hedging baseline (paper/docs/EXPERIMENT_PLAN_SIDE.md Section 14): one line of $LINES per
# array task, i.e. one family-task cell (its three member models) and one planner configuration
# (latency/ensemble.py).  Finished arms are skipped.
#   sbatch --array=0-15 --export=ALL,LINES=latency/runs_ensemble.txt slurm/latency_ens.sh
set -e
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYTHONPATH=${RCH_ROOT:-/mmfs1/storage/users/xiar3/exp/residual-calibrated-hedging}
export DMC_ENV_WORKERS=${DMC_ENV_WORKERS:-3}
export CKPT_KEEP=${CKPT_KEEP:-/mmfs1/storage/users/xiar3/exp/MetricTrust/checkpoints_keep}   # model checkpoints (not in this repo)
PY=/mmfs1/storage/users/xiar3/envs/metrictrust/bin/python
R=${RCH_ROOT:-/mmfs1/storage/users/xiar3/exp/residual-calibrated-hedging}
LINE=$(sed -n "$(( ${SLURM_ARRAY_TASK_ID:-0} + 1 ))p" $R/$LINES)
nvidia-smi --query-gpu=name --format=csv,noheader
echo "$LINE"
$PY $R/latency/ensemble.py $LINE

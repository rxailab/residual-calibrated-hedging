#!/bin/bash
#SBATCH --job-name=mt-lat-dmc
#SBATCH --partition=astro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --nice=100
#SBATCH --output=/scratch/hpc/11/xiar3/MetricTrust/runs/lat-dmc-%A_%a.out
# Side paper Studies 1-2 (sidepaper/EXPERIMENT_PLAN_SIDE.md): one line of $LINES per array task,
# i.e. one model and planner configuration, every arm of its preset (latency/run_dmc.py).
# Finished arms are skipped, so a repeated or requeued task never redoes work.
#   sbatch --array=0-26 --export=ALL,LINES=latency/runs_study1.txt slurm/latency_dmc.sh
#   sbatch --array=0-62 --time=04:00:00 --export=ALL,LINES=latency/runs_study2.txt slurm/latency_dmc.sh
# Runs on astro (L40 / L40S, same Ada architecture): every arm of a model is one job, so paired
# arms always share a GPU; the GPU is recorded with every result.  --nice keeps these behind any
# main-study job on the same partition.
set -e
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYTHONPATH=${RCH_ROOT:-/mmfs1/storage/users/xiar3/exp/residual-calibrated-hedging}
export DMC_ENV_WORKERS=${DMC_ENV_WORKERS:-6}
PY=/mmfs1/storage/users/xiar3/envs/metrictrust/bin/python
R=${RCH_ROOT:-/mmfs1/storage/users/xiar3/exp/residual-calibrated-hedging}
CK=${CKPT_KEEP:-/mmfs1/storage/users/xiar3/exp/MetricTrust/checkpoints_keep}   # model checkpoints (not in this repo)
LINE=$(sed -n "$(( ${SLURM_ARRAY_TASK_ID:-0} + 1 ))p" $R/$LINES)
nvidia-smi --query-gpu=name --format=csv,noheader
echo "$LINE"
$PY $R/latency/run_dmc.py $LINE

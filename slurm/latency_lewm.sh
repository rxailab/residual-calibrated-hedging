#!/bin/bash
#SBATCH --job-name=mt-lat-lewm
#SBATCH --partition=astro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --nice=100
#SBATCH --output=/scratch/hpc/11/xiar3/MetricTrust/runs/lat-lewm-%A_%a.out
# Study 15B (paper/docs/EXPERIMENT_PLAN_SIDE.md Section 15): LeWM delay schedules and RCH
# (latency/lewm.py).  One line of $LINES per array task: "<task> --arms ...".  Finished arms are skipped.
#   sbatch --array=0-N --export=ALL,LINES=latency/runs_lewm.txt slurm/latency_lewm.sh
set -e
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl SDL_VIDEODRIVER=dummy STABLEWM_HOME=${STABLEWM_HOME:-/scratch/hpc/11/xiar3/lewm}
export PYTHONPATH=${RCH_ROOT:-/mmfs1/storage/users/xiar3/exp/residual-calibrated-hedging}
PY=/mmfs1/storage/users/xiar3/envs/rch-lewm/bin/python
R=${RCH_ROOT:-/mmfs1/storage/users/xiar3/exp/residual-calibrated-hedging}
LINE=$(sed -n "$(( ${SLURM_ARRAY_TASK_ID:-0} + 1 ))p" $R/$LINES)
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo "$LINE"
$PY $R/latency/lewm.py $LINE

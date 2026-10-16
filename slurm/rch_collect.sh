#!/bin/bash
#SBATCH --job-name=rch-collect
#SBATCH --partition=astro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --nice=100
#SBATCH --output=/scratch/hpc/11/xiar3/MetricTrust/runs/rch-collect-%A_%a.out
# Section 16B: array index -> (task, shard), 40 shards of 50 episodes per new task, the main study's
# collection (experiments/dmc/collect.py) unchanged; data under $RCH_SCRATCH/data/dmc.
#   sbatch --array=0-119 slurm/rch_collect.sh
set -e
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl RCH_SCRATCH=/scratch/hpc/11/xiar3/RCH
R=/mmfs1/storage/users/xiar3/exp/residual-calibrated-hedging
export PYTHONPATH=$R
PY=/mmfs1/storage/users/xiar3/envs/metrictrust/bin/python
TASKS=(reacher-easy cheetah-run finger-spin)
N=50; SHARDS=40
T=${TASKS[$(( SLURM_ARRAY_TASK_ID / SHARDS ))]}; S=$(( SLURM_ARRAY_TASK_ID % SHARDS ))
OUT=$RCH_SCRATCH/data/dmc/$T/shard_$(printf %03d $S).npz
[ -f "$OUT" ] && { echo "exists: $OUT"; exit 0; }
$PY $R/experiments/dmc/collect.py --task $T --start $(( S * N )) --n $N --out $OUT

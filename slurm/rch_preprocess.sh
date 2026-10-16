#!/bin/bash
#SBATCH --job-name=rch-prep
#SBATCH --partition=serial
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/hpc/11/xiar3/MetricTrust/runs/rch-prep-%A_%a.out
# Section 16B: pack each new task's shards to 64 px (experiments/dmc/preprocess.py, unchanged).
#   sbatch --array=0-2 slurm/rch_preprocess.sh
export PYTHONNOUSERSITE=1 RCH_SCRATCH=/scratch/hpc/11/xiar3/RCH
R=/mmfs1/storage/users/xiar3/exp/residual-calibrated-hedging
TASKS=(reacher-easy cheetah-run finger-spin)
PYTHONPATH=$R /mmfs1/storage/users/xiar3/envs/metrictrust/bin/python $R/experiments/dmc/preprocess.py --task ${TASKS[$SLURM_ARRAY_TASK_ID]}

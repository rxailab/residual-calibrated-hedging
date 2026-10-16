#!/bin/bash
#SBATCH --job-name=rch-diag
#SBATCH --partition=astro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --nice=100
#SBATCH --output=/scratch/hpc/11/xiar3/MetricTrust/runs/rch-diag-%A_%a.out
# Section 16C: offline hedging value (latency/diagnose_anchor.py --extra) at d = 4 and 8, one model per
# array task from $LIST.  RCH_SCRATCH selects the data root (new tasks).
#   sbatch --array=0-23 --export=ALL,LIST=latency/runs_diag16_old.txt slurm/rch_diag.sh
set -e
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl
R=/mmfs1/storage/users/xiar3/exp/residual-calibrated-hedging
export PYTHONPATH=$R
CK=$(sed -n "$(( ${SLURM_ARRAY_TASK_ID:-0} + 1 ))p" $R/$LIST)
nvidia-smi --query-gpu=name --format=csv,noheader
/mmfs1/storage/users/xiar3/envs/metrictrust/bin/python $R/latency/diagnose_anchor.py $CK --extra --ds 4,8 --tag _v16

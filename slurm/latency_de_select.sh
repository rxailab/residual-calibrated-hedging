#!/bin/bash
#SBATCH --job-name=mt-lat-sel
#SBATCH --partition=serial
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/hpc/11/xiar3/MetricTrust/runs/lat-sel-%j.out
# Applies the pre-registered tuning rule (latency/select_lambda.py) after the tuning array,
# then launches the confirmatory array on the seed1/seed2 models.
set -e
export PYTHONNOUSERSITE=1 PYTHONPATH=${RCH_ROOT:-/mmfs1/storage/users/xiar3/exp/residual-calibrated-hedging}
R=${RCH_ROOT:-/mmfs1/storage/users/xiar3/exp/residual-calibrated-hedging}
CK=${CKPT_KEEP:-/mmfs1/storage/users/xiar3/exp/MetricTrust/checkpoints_keep}   # model checkpoints (not in this repo)
/mmfs1/storage/users/xiar3/envs/metrictrust/bin/python $R/latency/select_lambda.py
cd $R && sbatch --array=0-15 --time=10:00:00 --export=ALL,LINES=latency/runs_de_confirm.txt slurm/latency_dmc.sh

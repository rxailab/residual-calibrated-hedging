#!/bin/bash
#SBATCH --job-name=rch-train
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:tesla_v100-pcie-32gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --nice=100
#SBATCH --output=/scratch/hpc/11/xiar3/MetricTrust/runs/rch-train-%A_%a.out
# Section 16B: one line of $RUNS per array task.  TRAINER=train.py (RSSM, JEPA) or train_tdmpc2.py.
#   sbatch --array=0-17 --export=ALL,RUNS=latency/runs_newtasks_train.txt,TRAINER=train.py slurm/rch_train.sh
#   sbatch --array=0-8 --export=ALL,RUNS=latency/runs_newtasks_tdmpc2.txt,TRAINER=train_tdmpc2.py slurm/rch_train.sh
set -e
export PYTHONNOUSERSITE=1 RCH_SCRATCH=/scratch/hpc/11/xiar3/RCH
R=/mmfs1/storage/users/xiar3/exp/residual-calibrated-hedging
export PYTHONPATH=$R
ARGS=$(sed -n "$(( SLURM_ARRAY_TASK_ID + 1 ))p" $R/$RUNS)
nvidia-smi --query-gpu=name --format=csv,noheader
echo "$ARGS"
/mmfs1/storage/users/xiar3/envs/metrictrust/bin/python $R/experiments/dmc/$TRAINER $ARGS

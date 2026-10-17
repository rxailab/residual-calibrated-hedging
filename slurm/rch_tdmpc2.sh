#!/bin/bash
#SBATCH --job-name=rch-td2
#SBATCH --partition=astro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --nice=100
#SBATCH --output=/scratch/hpc/11/xiar3/MetricTrust/runs/rch-td2-%A_%a.out
# Study 17B (EXPERIMENT_PLAN_SIDE.md Section 17): released TD-MPC2 models under a pipelined delay
# (latency/tdmpc2_released.py).  One line of $LINES ("<task> <seed> [flags]") per array task.
#   sbatch --array=0-16 --export=ALL,LINES=latency/runs_tdmpc2_released.txt slurm/rch_tdmpc2.sh
set -e
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl RCH_SCRATCH=/scratch/hpc/11/xiar3/RCH
R=/mmfs1/storage/users/xiar3/exp/residual-calibrated-hedging
export PYTHONPATH=$R
LINE=$(sed -n "$(( ${SLURM_ARRAY_TASK_ID:-0} + 1 ))p" $R/$LINES)
nvidia-smi --query-gpu=name --format=csv,noheader
echo "$LINE"
/mmfs1/storage/users/xiar3/envs/metrictrust/bin/python $R/latency/tdmpc2_released.py $LINE

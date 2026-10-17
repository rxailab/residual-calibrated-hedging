#!/bin/bash
#SBATCH --job-name=mt-lat-wall4
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:tesla_v100-pcie-32gb:1
#SBATCH --exclusive
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --nice=100
#SBATCH --output=/scratch/hpc/11/xiar3/MetricTrust/runs/lat-wall4-%A_%a.out
# Section 19 (EXPERIMENT_PLAN_SIDE.md): real-time htr, cpp and rch on the 21 models of the 7 cells whose
# fixed-delay hedge-then-refine gain is at least 0.03; one exclusive V100 node per model.
#   sbatch --array=0-20 slurm/latency_wallclock_v4.sh
set -e
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl RCH_SCRATCH=/scratch/hpc/11/xiar3/RCH
R=/mmfs1/storage/users/xiar3/exp/residual-calibrated-hedging
export PYTHONPATH=$R
CK=$(sed -n "$(( ${SLURM_ARRAY_TASK_ID:-0} + 1 ))p" $R/latency/runs_wallclock_v4.txt)
nvidia-smi --query-gpu=name --format=csv,noheader
echo "$CK"
/mmfs1/storage/users/xiar3/envs/metrictrust/bin/python $R/latency/wallclock.py $CK --arms htr,cpp,rch --episodes 50 \
  --calibrate --seed0 950200 --out-sub wallclock_v4

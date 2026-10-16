#!/bin/bash
#SBATCH --job-name=mt-lat-wall2
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:tesla_v100-pcie-32gb:1
#SBATCH --exclusive
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --nice=100
#SBATCH --output=/scratch/hpc/11/xiar3/MetricTrust/runs/lat-wall2-%A_%a.out
# Study 15C (EXPERIMENT_PLAN_SIDE.md Section 15): real-time cpp, rch and aligned on the six cup-catch
# models of JEPA and TD-MPC2, d_est calibrated from one unscored real-time episode (latency/wallclock.py
# --calibrate), one exclusive V100 node per model.
#   sbatch --array=0-5 slurm/latency_wallclock_v2.sh
set -e
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYTHONPATH=${RCH_ROOT:-/mmfs1/storage/users/xiar3/exp/residual-calibrated-hedging}
PY=/mmfs1/storage/users/xiar3/envs/metrictrust/bin/python
R=${RCH_ROOT:-/mmfs1/storage/users/xiar3/exp/residual-calibrated-hedging}
CK=${CKPT_KEEP:-/mmfs1/storage/users/xiar3/exp/MetricTrust/checkpoints_keep}
M=(jepa/cup-catch/seed0/step0060000 jepa/cup-catch/seed1/step0060000 jepa/cup-catch/seed2/step0060000
   tdmpc2/cup-catch/seed0/step0100000 tdmpc2/cup-catch/seed1/step0100000 tdmpc2/cup-catch/seed2/step0100000)
nvidia-smi --query-gpu=name --format=csv,noheader
$PY $R/latency/wallclock.py $CK/dmc/${M[${SLURM_ARRAY_TASK_ID:-0}]}.pt --arms rch,cpp,aligned --episodes 30 --calibrate

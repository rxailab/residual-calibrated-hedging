#!/bin/bash
#SBATCH --job-name=rch-agents
#SBATCH --partition=astro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/hpc/11/xiar3/MetricTrust/runs/rch-agents-%j.out
# Section 16B: released TD-MPC2 agent check on the new tasks (experiments/dmc/agent_check.py).
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl
R=/mmfs1/storage/users/xiar3/exp/residual-calibrated-hedging
for T in reacher-easy cheetah-run finger-spin; do PYTHONPATH=$R /mmfs1/storage/users/xiar3/envs/metrictrust/bin/python $R/experiments/dmc/agent_check.py $T; done

#!/bin/bash
#SBATCH --job-name=mt-lat-time
#SBATCH --exclusive
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --nice=100
#SBATCH --output=/scratch/hpc/11/xiar3/MetricTrust/runs/lat-time-%j.out
# Side paper Study 2 timing: single-environment call latency (latency/timing.py) of the nine
# seed0 latent models, one after another, on a whole node of the requested GPU type, so no other
# job shares its GPUs, CPUs or PCIe.  DIFF=1 also times the diffusion planner (V100 only, to match
# the configuration records of Fig. 2).
#   sbatch -p gpu-short --gres=gpu:tesla_v100-pcie-32gb:1 --export=ALL,DIFF=1 slurm/latency_timing.sh
#   sbatch -p astro --gres=gpu:nvidia_l40s:1 slurm/latency_timing.sh
#   sbatch -p gpu-short --gres=gpu:nvidia_h200_nvl:1 slurm/latency_timing.sh
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYTHONPATH=${RCH_ROOT:-/mmfs1/storage/users/xiar3/exp/residual-calibrated-hedging}
PY=/mmfs1/storage/users/xiar3/envs/metrictrust/bin/python
R=${RCH_ROOT:-/mmfs1/storage/users/xiar3/exp/residual-calibrated-hedging}
CK=${CKPT_KEEP:-/mmfs1/storage/users/xiar3/exp/MetricTrust/checkpoints_keep}   # model checkpoints (not in this repo)
nvidia-smi --query-gpu=name --format=csv,noheader
rc=0
while read -r CK; do
  $PY $R/latency/timing.py $CK || rc=1
done < $R/latency/runs_timing.txt
if [ -n "$DIFF" ]; then
  $PY $R/latency/run_diffusion.py $CK/dmc/diffusion/dmcontrol/seed0/step0032000.pt \
    --task cartpole-swingup --timing || rc=1
fi
exit $rc

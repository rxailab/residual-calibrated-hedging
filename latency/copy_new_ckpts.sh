#!/bin/bash
# Section 16B: copy the final checkpoints of the 27 new-task models from scratch to storage
# (scratch is purged by access time); evaluation reads the storage copies.
set -e
S=/scratch/hpc/11/xiar3/RCH/checkpoints/dmc
K=/mmfs1/storage/users/xiar3/exp/residual-calibrated-hedging/checkpoints_new/dmc
for t in reacher-easy cheetah-run finger-spin; do
  for f in rssm jepa tdmpc2; do
    st=step0060000.pt; [ $f = tdmpc2 ] && st=step0100000.pt
    for s in 0 1 2; do
      src=$S/$f/$t/seed$s/$st
      if [ -f "$src" ]; then mkdir -p $K/$f/$t/seed$s && cp -n $src $K/$f/$t/seed$s/; else echo "missing $src"; fi
    done
  done
done

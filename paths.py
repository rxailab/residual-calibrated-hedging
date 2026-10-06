"""Canonical paths and the storage policy.

Two filesets, with different guarantees (observed 2026-09-26):

* storage  /mmfs1/storage/users/xiar3   200 GB quota, NOT purged.
  Holds code, the conda env, results, the registry, and every small artifact
  needed to regenerate what lives on scratch (seeds, data-agent policies).
* scratch  /scratch/hpc/11/xiar3        10 TB quota, PURGED by access time
  (files unread for about three weeks are deleted).
  Holds only regenerable data and disposable checkpoints.

Consequence for the pipeline: every utility and metric value is written to
storage as soon as a checkpoint is evaluated, so a purged checkpoint costs a
retrain, never a result.  Datasets are a deterministic function of seeds and
data-agent policies stored on storage.
"""
from pathlib import Path
import os

PROJECT = Path(__file__).resolve().parent
ENV = Path("/mmfs1/storage/users/xiar3/envs/metrictrust")          # storage
PYTHON = ENV / "bin" / "python"
SCRATCH = Path(os.environ.get("METRICTRUST_SCRATCH", "/scratch/hpc/11/xiar3/MetricTrust"))

DATA = SCRATCH / "data"              # regenerable from seeds + policies
CHECKPOINTS = SCRATCH / "checkpoints"  # disposable once evaluated
RUNS = SCRATCH / "runs"              # slurm logs
RESULTS = PROJECT / "results"        # storage: every number the paper uses
UTILITY_STORE = RESULTS / "utility"
METRIC_STORE = RESULTS / "metrics"
REGISTRY = PROJECT / "registry"      # storage: checkpoint metadata
ARTIFACTS = PROJECT / "artifacts"    # storage: small regeneration inputs

for _p in (DATA, CHECKPOINTS, RUNS, UTILITY_STORE, METRIC_STORE, REGISTRY, ARTIFACTS):
    _p.mkdir(parents=True, exist_ok=True)

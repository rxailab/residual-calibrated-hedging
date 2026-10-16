# Residual-calibrated hedging for planning with learned world models under inference delay

This is the paper, code and results of the side paper. It was split out of the private MetricTrust repository (`rxailab/MetricTrust`) on 6 October 2026, at commit `49f98a2`.

**The method.** A planner that uses a learned world model takes time to compute, and a deployed system keeps moving while it does. Committed-prefix planning rolls the model through the actions already committed and plans from the predicted post-delay state. With learned pixel world models this leaves most of the delay loss, because the models' errors through the delay are value-aligned.

Residual-calibrated hedging (RCH) scores each candidate over several post-delay states: the prediction plus the model's own recorded d-step errors. These states are scored in one batched call, which adds no latency at batch size 1. Optionally, a small learned heteroscedastic error model replaces the recorded errors.

**Latest paper:** [PRX Intelligence manuscript (PDF)](paper/prxintelligence/build/main.pdf) and [Supplemental Material (PDF)](paper/prxintelligence/build/supplementary.pdf), prepared with APS REVTeX 4.2. [LaTeX sources and build instructions](paper/prxintelligence/).

## Layout

| Path | Contents |
|---|---|
| `paper/prxintelligence/` | Latest manuscript: `main.tex`, `main-review.tex`, `refs.bib` and `supplementary.tex` (APS REVTeX 4.2), with built PDFs in `build/` |
| `paper/ncomms/` | Previous Nature-format manuscript: `main.tex`, `refs.bib` and `supplementary.tex` (Springer Nature template v3.1, `sn-nature`), with built PDFs in `build/` |
| `paper/figures/`, `paper/scripts/` | Figures 1-3 (SVG, PDF and PNG) and the scripts that generate them from data (`figstyle.py` plus one script per figure) |
| `paper/fonts/` | Albert Sans (SIL OFL, `OFL.txt`) and `fonts.conf` for the figure renderer |
| `paper/evidence/`, `paper/analyse_pilot.py` | The diffusion planner's configuration records behind Fig. 1, and their reanalysis |
| `paper/docs/` | Experiment plan with pre-registered rules and launch log (`EXPERIMENT_PLAN_SIDE.md`); novelty audit (`NOVELTY_AUDIT.md`) |
| `latency/` | The method and studies (table below) |
| `tests/test_latency.py` | Gate G0: committed-prefix planning equals paused planning that replans every d decisions under an exact model; equal compute; shift invariance |
| `slurm/latency_*.sh` | Job scripts for the Lancaster HEC cluster |
| `results/latency/` | All results as JSON. Cached error-model weights are not included; they are regenerated on first use |
| `wm/`, `planners/`, `envs/`, `metrics/`, `experiments/dmc/`, `paths.py` | MetricTrust modules the code imports (world-model wrappers, MPPI, DMC environment, evaluators), copied unchanged |
| `third_party/shims/` | Attention shim used by the diffusion wrapper |

**The `latency/` package:**

| File | What it does |
|---|---|
| `schedule.py` | Planning schedules (paused, naive, time-aligned, committed-prefix) and `PrefixMPPI` |
| `hedge.py` | Residual-calibrated hedging |
| `delay_error.py` | The learned delay-error model and the ablations |
| `ensemble.py` | Baselines: DATS-style ensemble hedging over the three seed models of a cell, and particle anchors from the RSSM's stochastic transition |
| `run_dmc.py`, `run_diffusion.py` | Closed-loop drivers |
| `timing.py`, `timing_hedge.py`, `timing_ensemble.py` | Latency measurements (`timing_ensemble.py` also records memory and parameters) |
| `wallclock.py` | The real-time check |
| `diagnose_anchor.py`, `error_model.py` | Offline diagnostics |
| `analyse*.py`, `select_lambda.py` | Pre-registered analyses (`analyse_ensemble.py`: the ensemble and particle baselines) |
| `runs_*.txt` | The run lists that were executed |

## Build the paper

Run from `paper/prxintelligence/`:

```bash
tectonic --outdir build main.tex
tectonic --outdir build supplementary.tex
```

The manuscript reuses `paper/figures/` through its graphics search path. Use `main-review.tex` for the optional single-column version. See `paper/prxintelligence/README.txt` for pdfLaTeX/BibTeX instructions.

The previous Nature-format version remains in `paper/ncomms/`; build it there with `tectonic -X compile -Z search-path=.. --outdir build main.tex`. On the HEC cluster, tectonic is `/mmfs1/storage/users/xiar3/envs/texenv/bin/tectonic`.

Regenerate a figure from `paper/` (the env is a venv with cairosvg and Pillow):

```bash
/mmfs1/storage/users/xiar3/envs/figenv/bin/python scripts/fig_delay_curves.py
```

`fig_motivation.py` and `fig_timeline.py` build the other two figures the same way.

## Run the experiments

The code runs in the MetricTrust conda environment (`/mmfs1/storage/users/xiar3/envs/metrictrust`, with `PYTHONNOUSERSITE=1`).

It needs inputs from the MetricTrust pipeline that are **not in this repo**:

- **DMControl datasets:** pre-processed under `/scratch/hpc/11/xiar3/MetricTrust/data` (`dmc64`, `dmc_lat`).
- **Diffusion reward heads:** `results/diffusion/heads` in MetricTrust.
- **Trained world models:** the final checkpoints in `/mmfs1/storage/users/xiar3/exp/MetricTrust/checkpoints_keep`, which the Slurm scripts find through `CKPT_KEEP`.
- **Diffusion model only:** MiniWorld (github.com/Zhao-Yian/MiniWorld at `e484206`), placed in `third_party/MiniWorld`.

The scripts write results inside this repo; override the root with `RCH_ROOT`.

```bash
PYTHONNOUSERSITE=1 python tests/test_latency.py
```

```bash
sbatch --array=0-26 --export=ALL,LINES=latency/runs_study1.txt slurm/latency_dmc.sh
```

The first command is the CPU exactness check. The second runs Study 1 (delay sweep); the other studies follow the same pattern with their `runs_*.txt`. See `paper/docs/EXPERIMENT_PLAN_SIDE.md` for every study, its decision rule and its job IDs.

## Status

The latest manuscript is in PRX Intelligence format (16 October 2026), including the comparison with DATS-style ensemble hedging (11 October 2026; `latency/ensemble.py`, `results/latency/ensemble.json`). Still to fill in: archive and code DOIs, and the supplement's existing per-study compute-total placeholder.

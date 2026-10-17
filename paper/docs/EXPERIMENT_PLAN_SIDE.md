# Side paper: experiment plan

**Paper.** *Planning through latency: committed-prefix control and latency-faithful evaluation for slow world models* (Nature Communications Article format, `ncomms/main.tex`).
**Version.** 0.1, 4 October 2026, written before any side-paper run. Nothing below has been run.
**Compute.** About 155 V100-hours plus about 4 hours of L40S/H200 time for timing only (Section 6). No training.

---

## 1. What the paper must show

The paper proposes a method, not a measurement report. Its claims, and the evidence each needs:

| Claim | Evidence needed | Study |
|---|---|---|
| C1. Paused-simulator evaluation hides latency, and amortised cost is not response latency. | Existing diffusion configuration records (already in `evidence/`), plus single-environment latencies measured properly. | 0 (done), 2 |
| C2. Committed-prefix planning (CPP) is exact for a perfect model. | Proof (in Methods) and a bit-for-bit numerical check. | G0 |
| C3. CPP recovers most of the return lost by time-aligned asynchronous planning, at no extra compute. | Delay sweep over three families, three tasks, three models each. | 1 |
| C4. The delay decomposition separates feedback-rate loss from mismatch loss and is informative. | Paused arm at r = d alongside the asynchronous arms. | 1 |
| C5. Paused rankings of planner configurations do not survive real latency; the latency-aware rule recovers return. | Latency of 8 configurations on 3 GPU types; returns across the delay grid. | 2 |
| C6. The method generalises to models we did not train and to a diffusion world model. | Public LeWM checkpoints (3 tasks); MiniWorld diffusion model (2 tasks). | 3 |
| C7. The delay-in-decisions abstraction reproduces wall-clock asynchronous execution. | Threaded real-time runs compared with fixed-d runs. | 4 |

## 2. Separation from the main MetricTrust paper

Used here: the final checkpoints of the RSSM, JEPA and TD-MPC2 runs (seed0, seed1, seed2) and the diffusion model's 32k seed0 snapshot, the DMControl data pipeline, the MPPI implementation, and the 18 diffusion configuration records (timings only).

Not used here, and must not appear in the paper: any offline metric (M1-M12), validity or regret results, checkpoint rankings or between-family quality comparisons, within/between-run populations, ROF, latent-space normalisation, the TwoRoom cost-swap result, the LunarLander results, and the DINO-WM / LIBERO held-out sets (reserved for the main paper's confirmatory test). The paper never uses intermediate snapshots, so it cannot be read as a checkpoint-selection study.

Episodes use **new initial-state seeds 950,000-950,049**, disjoint from S_U1 (900,000-900,099) and the LunarLander sets (800,000-, 900,000-), so no closed-loop number in the side paper duplicates a main-paper number. Check with `grep -rn "950_0\|9500" experiments scripts` before the runs.

The public LeWM checkpoints are run on PushT, Reacher and Cube only. TwoRoom stays with the main paper.

## 3. Implementation (new files only)

The main diffusion evaluation (array 24927064) is still starting tasks; its pending tasks import `experiments/dmc/evaluate.py`, `evaluate_diffusion.py` and `planners/mppi.py` at start. **Do not edit those files until the array has finished.** All side-paper code goes in a new package `latency/` that imports from them.

| File | Contents |
|---|---|
| `latency/schedule.py` | The delay-in-decisions episode loop, family-agnostic: arms `paused(r)`, `naive(d)`, `aligned(d)`, `cpp(d)`; pre-episode plan with the simulator paused; candidate length d + H in asynchronous arms; warm start shifted by the decisions executed. |
| `latency/mppi_prefix.py` | Batched MPPI step with an optional committed prefix (overwrite positions 0:d of every candidate after clipping); otherwise identical to `evaluate.mppi_step`. |
| `latency/run_dmc.py` | Study 1 and 2 driver for RSSM / JEPA / TD-MPC2 (reuses `EnvPool`, adapters, `filter_step`); writes one JSON per (family, task, seed, arm, d, config) with per-episode returns, decisions, plan time, non-finite counts and the git hash. |
| `latency/run_diffusion.py` | Study 3 diffusion driver, d in whole blocks (8, 16 decisions); prefix in blocks. |
| `latency/run_lewm.py` | Study 3 LeWM driver in the stable-worldmodel env (`PYTHONNOUSERSITE=1`, `/mmfs1/storage/users/xiar3/envs/metrictrust-swm`), native CEM with prefix overwrite. |
| `latency/timing.py` | Single-environment call latency: 20 warm-up calls, then 200 timed calls with `torch.cuda.synchronize()` at both ends, states sampled from evaluation episodes; records GPU name, clocks (`nvidia-smi -q -d CLOCK`), other processes on the GPU. |
| `latency/wallclock.py` | Study 4: planner thread plus a simulator thread paced by `time.monotonic()` at Δt; the action queue is read at each tick; records the realised delay of every call. |
| `tests/test_latency.py` | G0 exactness test and conformance checks (Section 4). |
| `slurm/latency_*.sh` | One array per study; each task loads its config line from `latency/runs_*.txt`. |

Copy the 27 final latent checkpoints (about 1 GB) and the diffusion 32k seed0 snapshot from scratch to `/mmfs1/storage/users/xiar3/exp/MetricTrust/checkpoints_keep/` first: scratch is purged by access time (plan amendment A6).

## 4. Gate G0: exactness and conformance (CPU or one GPU, under 1 hour)

1. **Exactness.** In the linear-Gaussian testbed (`utility/u1_planning.py`) with process noise off and the true system as the model, run `paused(d)` and `cpp(d)` for d in {1, 2, 4} with the same noise stream per call. Pass: identical action sequences and returns to floating-point tolerance (1e-9). Also confirm that `aligned(d)` differs.
2. **Equivalence at d = 0.** `cpp` with an empty prefix reproduces `evaluate.mppi_step` bit for bit on one RSSM checkpoint (TINY settings).
3. **Equal compute.** `aligned(d)` and `cpp(d)` make the same number of model calls with the same tensor shapes (assert in the driver).
4. **Prefix invariance.** For a fixed batch of candidates, adding a constant to all costs leaves the MPPI weights unchanged (unit test).

No study starts until G0 passes.

## 5. Studies

### Study 0: motivating reanalysis (done)

`analyse_pilot.py` on `evidence/pilot_records.json`; Fig. 2 from `scripts/fig_motivation.py`. Only timings are used in the paper.

### Study 1: delay robustness and decomposition (primary)

- **Models.** RSSM, JEPA, TD-MPC2; cartpole-swingup, walker-walk, cup-catch; final checkpoints of seed0, seed1, seed2: 27 models.
- **Floor rule (fixed now).** A (family, task) cell enters pooled estimates only if its mean `paused(1)` normalised return over the three models is at least 0.15. Expected exclusion: JEPA walker (0.03-0.06 on the main-study seeds).
- **Arms.** `paused(1)`; and for each d in {1, 2, 4, 8, 16}: `paused(d)`, `naive(d)`, `aligned(d)`, `cpp(d)`. 21 runs per model.
- **Planner.** Population 512, 6 iterations, H = 32, temperature 0.5 (main-study settings), candidate length 32 + d in asynchronous arms.
- **Episodes.** 50 per run (seeds 950,000-950,049), shared across arms. Sample size check: on the main grid the per-episode SD of normalised return is about 0.1-0.3; a paired SD of 0.2 gives a standard error of 0.028 per model-arm contrast and about 0.006 pooled over 24 models, enough to resolve a pooled CPP-minus-aligned difference of 0.03. Re-check with the seed0 runs (pilot, below) before the rest.
- **Pilot.** seed0 of every family and task first (9 models, about 12 V100-hours). If the pooled paired SD is above 0.35, raise episodes to 100 for the remaining models (adds about 20 V100-hours).
- **Cost.** About 1.4 V100-hours per model at 50 episodes (batched lockstep episodes; planning time falls roughly as 1/d). 27 models: **about 38 V100-hours**.

### Study 2: latency-aware configuration

- **Timing.** For each family (seed0 models) and each configuration c in population {64, 128, 256, 512} x iterations {2, 6}, measure L50 and L95 at batch size 1 on a V100, an L40S and an H200 (8 configurations x 3 families x 3 GPUs = 72 timing jobs of about 2 minutes; about 3 hours in all). Also time the diffusion planner (s in {2, 4, 8}, population 64) and LeWM's CEM on the V100 at batch 1, which gives the exact single-environment figure the paper's motivation section is waiting for.
- **Returns.** For each family x task (seed0 models) x configuration: `paused(1)` and `cpp(d)` for d in {1, 2, 4, 8, 16} (a grid that covers every measured d(c), so any GPU's operating point is read off the grid without new runs). 3 x 3 x 8 x 6 = 432 runs at 50 episodes; cheap configurations dominate. **About 30 V100-hours.**
- **Outputs.** Kendall's tau between the paused ranking and the latency-faithful ranking of the 8 configurations per (family, task, GPU); return of c* minus the paused-optimal configuration at its own delay.

### Study 3: generality

- **LeWM public checkpoints** (`quentinll/lewm-pusht`, `lewm-reacher`, `lewm-cube`) with stable-worldmodel's evaluation harness and native CEM. d in {1, 2, 4} action blocks (LeWM plans in blocks of 5 steps); arms `paused(1)`, `paused(d)`, `aligned(d)`, `cpp(d)`; 100 episodes per run (binary success needs more episodes). First check that each task's `paused(1)` success reproduces the published value within its Wilson interval; a task that fails is dropped and reported. **About 15 V100-hours** (estimate; measure on the first task).
- **Diffusion world model** (32k seed0 snapshot, 2 sampler steps). Cartpole and cup-catch only (walker is at the floor, 0.02-0.05). Arms `paused(8)`, `paused(16)`, `aligned(8)`, `cpp(8)`, `aligned(16)`, `cpp(16)`; 10 episodes per run in two batches of 5. From the configuration records, about 2.1 V100-hours per 5 episodes at r = 8 and about half at r = 16, times 1.25 (d = 8) or 1.5 (d = 16) for the longer candidates: about 23 V100-hours per task. **About 46 V100-hours.** Reported descriptively (10 episodes per arm); no pooled hypothesis test uses these runs.

### Study 4: wall-clock faithfulness

RSSM seed0 on cartpole-swingup (Δt = 0.02 s), single environment, one V100, planner configurations whose measured L95 gives d of about 2 and about 4. 30 wall-clock episodes per configuration with `cpp` and `aligned`, compared with Study 1/2 runs at d = ceil(L50/Δt) and ceil(L95/Δt) on the same seeds. Rendering runs on CPU in the simulator thread; the GPU node must be otherwise idle (request `--exclusive`). **About 3 V100-hours.**

## 6. Budget and wall-clock

| Study | V100-hours | Other GPU | Wall-clock on available V100s |
|---|---|---|---|
| G0 tests | 1 | - | half a day including writing the code |
| 1 delay sweep | 38 (+20 if episodes rise to 100) | - | about 4 h on 12 V100s |
| 2 configuration | 30 | 3 h timing (L40S, H200) | about 4 h on 12 V100s; timing jobs whenever an H200 / L40S frees |
| 3 LeWM | 15 | - | about 2 h |
| 3 diffusion | 46 | - | about 3 h (24 jobs of 1-2.7 h, 5 episodes each) |
| 4 wall-clock | 3 | - | 3 h, one exclusive node |
| Contingency 15 % | 20 | | |
| **Total** | **about 155** | **about 4 h** | **about 2 days after the code passes G0** |

V100s are tied up by the main diffusion evaluation until about 11:00 on 5 October; side-paper jobs should go to gpu-short behind it and never delay main-study jobs.

## 7. Pre-specified hypotheses (Supplementary Note 1)

Primary (Holm-corrected as one family of three):

- **H1.** Pooled over cells that pass the floor rule, R(cpp(d)) - R(aligned(d)) > 0 at each of d = 2, 4, 8. Test: one-sided hierarchical bootstrap, alpha = 0.05 after Holm. Falsified if the interval includes 0 at any of the three delays.
- **H2.** At d = 4 and d = 8, CPP removes at least half of the time-aligned planner's mismatch loss: M_cpp(d) <= 0.5 M_aligned(d), pooled. Falsified if the bootstrap interval of M_cpp(d) / M_aligned(d) lies above 0.5.
- **H3.** For at least 5 of the 9 (family, task) cells on the V100, the latency-aware configuration c* differs from the paused-optimal configuration and has higher return at its own delay. Falsified if fewer than 5.

Secondary (estimates with intervals, no test): the decomposition per family and d; naive vs aligned; Kendall's tau of paused vs latency-faithful configuration rankings per GPU; LeWM and diffusion results; wall-clock agreement.

A negative outcome is still reportable: if H1 fails, the paper becomes a protocol-and-decomposition paper whose decomposition explains why prefix prediction did not help (large M_cpp means the model cannot predict through the delay).

## 8. Analysis

`latency/analyse.py`: reads the run JSONs, applies the floor rule, computes per-model means and paired differences, the hierarchical bootstrap (models within cell, episodes within model; 10,000 resamples; seed fixed), the decomposition terms, Holm adjustment, and writes `results/latency/summary.json` and Figs 3-5. Written and frozen (git commit) before Study 1's non-pilot runs finish.

## 9. Risks

| Risk | Mitigation |
|---|---|
| Scratch purge removes final checkpoints | Copy to storage before anything else (Section 3). |
| Main diffusion evaluation still running | New files only; side jobs queued behind it on gpu-short. |
| CPP shows no gain because the latent models predict poorly through d | H2 and the decomposition make this a measured result; report M_cpp(d) per family. |
| LeWM harness does not reproduce published success on a task | Drop that task, report it. |
| Diffusion runs too noisy at 10 episodes | Descriptive only; the paper's claims rest on Studies 1-2. |
| Real latency jitter makes fixed d unrepresentative | Study 4 measures it; report d from L95 and L50. |

## 10. Order of work

1. Copy checkpoints to storage. Write `latency/` and the G0 tests; pass G0.
2. Study 2 timing (cheap, informs which d values matter) and Study 1 pilot (seed0).
3. Freeze `latency/analyse.py`; run the rest of Study 1, Study 2 returns, Study 3, Study 4.
4. Fill the `\pending{}` items in `ncomms/main.tex`; write Figs 3-5 and Table 2.

---

## 11. Launch log

**4 October 2026, 22:50-23:20.** Code in `latency/` (schedule.py, run_dmc.py, timing.py, run_diffusion.py, wallclock.py), Slurm scripts `slurm/latency_*.sh`, and run lists `latency/runs_*.txt`.

**Gate G0 passed** (`python tests/test_latency.py`, CPU). With an exact linear model, cpp(d) reproduces paused(d)'s actions to 1e-9 for d = 1, 2 and 4, and aligned and naive differ. An inexact model breaks the equality. Aligned and cpp make identical cost calls, and MPPI is invariant to a constant cost shift. G0 item 2 (bit equality with `evaluate.mppi_step`) was dropped. `PrefixMPPI` gives non-finite costs weight zero (D11) where `mppi_step` maps them to 1e9. The main study recorded no non-finite costs for these models.

**Smoke tests passed.** Real EGL rendering on astro: RSSM paused1 and cpp4 at 0.64 s per call for 50 batched episodes on an L40S. Diffusion: all arms with a tiny planner on an L40. Timing script: TD-MPC2 cup-catch at 25-115 ms per single-environment call on an L40S.

| Study | Job | Where | Notes |
|---|---|---|---|
| 1 | 24931612, array 0-26 | astro (L40 / L40S) | seed0 models first (tasks 0-8) |
| 3, diffusion | 24931613, array 0-23 | astro (L40 / L40S) | 2 tasks x 2 seed batches x 6 arms, 5 episodes each |
| 2, returns | 24931614, array 0-62 | astro (L40 / L40S) | 7 configurations x 9 models; pop 512 / 6 iterations come from Study 1 |
| 2, timing V100 (+ diffusion latency) | 24931680 | gpu-short, exclusive node | |
| 2, timing L40S | 24931681 | astro, exclusive node | |
| 2, timing H200 | 24931682 | gpu-short, exclusive node | |
| 4, wall clock | 24931683, array 0-1 | gpu-short V100, exclusive node | pop 512 / 6 iterations and pop 128 / 2 iterations |

**Deviations from Sections 5-6, decided before any result:**

1. **Closed-loop returns run on astro L40 / L40S, not V100s.** The V100s are held by the main diffusion evaluation until about 11:00 on 5 October, and astro was idle. Every arm of a model runs in one job, so paired arms always share a GPU. Every diffusion job is restricted to astro, which has one GPU architecture (Ada), so all diffusion arms share sampler precision. Latency is measured separately per GPU type, so the choice of GPU for returns does not enter any latency figure.
2. **Timing jobs run the nine models in sequence on an exclusive node, one job per GPU type**, instead of 72 separate jobs, so that co-located jobs cannot inflate L95. Diffusion single-environment latency is measured on the V100 only, on cartpole (s = 2, 4, 8; d = 0, 8, 16; 5 timed calls each).
3. **The LeWM part of Study 3 is not launched.** Only the TwoRoom LeWM checkpoint and dataset exist locally. PushT, Reacher and Cube need the public checkpoints and datasets downloaded, and a harness that reproduces each task's published success rate first (Section 5, Study 3).

Results: `results/latency/{dmc,diffusion,timing,wallclock}/`. Logs: `/scratch/hpc/11/xiar3/MetricTrust/runs/lat-*.out`.

---

## 12. Method search after the novelty audit (option B, 5 October 2026)

`sidepaper/NOVELTY_AUDIT.md` showed that the v2 method claims are not novel. Option B looks for a method that reduces the residual mismatch M_cpp(d) left by committed-prefix planning. A second prior-art search covered four candidates. Sensitivity correction (C1) is close to asNMPC and F-MPPI. Multi-anchor selection (C2) is close to multi-stage/contingency MPC, DATS and possibly CLIS-MPPI (CDC 2026, content not public). Online calibration (C3) is classical. Re-anchoring between iterations (C4) matches Melon et al. 2021 and RTI.

**Offline diagnostic** (`latency/diagnose_anchor.py`, D_eval windows, model-predicted cost from the true post-delay latent; results in `results/latency/diagnose/`):

- The predicted anchor is close to the truth: its error is 1-20 % of the stale error for RSSM and JEPA, and 7-140 % for TD-MPC2.
- Where committed-prefix planning fails in closed loop (RSSM walker, JEPA, TD-MPC2), the predicted anchor still causes systematic regret.
- The actual error costs 1.3-3x more than a random error of the same size. Halving the error roughly halves the regret.
- Corrections that do not work: C1 (finite-difference sensitivity on the top 8 residual principal components, which hold only 10-50 % of the error), bias correction, and C2 nearest-anchor selection.
- What works: hedging, i.e. averaging each candidate's cost over K anchors (the predicted anchor plus calibration residuals). At d = 2, K = 16 removes 25-55 % of the regret where there is any, and changes nothing where there is none.

**Candidate method.** Residual-hedged committed-prefix MPPI (`latency/hedge.py`, arms `hcpp<d>k<K>`). The prefix is rolled once. The free part is scored from K anchors: z_hat plus K - 1 residuals drawn per episode and call from a bank of 512 d-step residuals of the same model on D_train windows. The anchor-averaged cost replaces the cost. K = 1 is committed-prefix planning in its roll-once form. It needs no ensemble and no retraining, so it works for single deterministic world models.

**Closed-loop pilot, decision rule written before the runs.** The 8 seed0 models of the eligible cells, population 512, 6 iterations, the 50 Study 1 seeds. Arms hcpp{d}k1, hcpp{d}k4 and hcpp{d}k16 at d = 2, 4 and 8; paused(d), aligned(d) and cpp(d) come from Study 1. The primary pilot contrast is R(hcpp_k16) - R(hcpp_k1), pooled over cells with equal weight, at each d, with an episode bootstrap (one model per cell).

- **Proceed** to the full study (3 seeds, latency-matched comparison, diffusion) if the contrast is positive with its 95 % interval above 0 at two or more of the three delays.
- **Report the method as ineffective** and fall back to option A if the contrast's interval includes 0 at all three delays.
- Otherwise, discuss it with the author.

Hedging multiplies the planner's compute by K. Whether that costs latency at batch size 1 is a separate question, answered by the Study 2 timing jobs. The pilot tests efficacy at equal population, not at equal latency.

**Pilot result (5 October 2026, 13:25; `results/latency/hedge_pilot.json`).** R(k16) - R(k1), pooled over 8 seed0 cells:

| d | estimate | 95 % interval | Share of remaining mismatch removed |
|---|---|---|---|
| 2 | -0.017 | [-0.041, +0.005] | none |
| 4 | +0.022 | [+0.0003, +0.044] | 0.26 |
| 8 | +0.045 | [+0.015, +0.076] | 0.45 |

By the rule the verdict is **proceed**, but at d = 4 only by a hair. With 50 episodes batched, a K = 16 call costs 16.8x the time of K = 1 (range 15.4-17.3), so the pilot is an efficacy test at equal population, not at equal latency. A batch-1 latency test of hedged calls was submitted (job 24946728).

**Replication, written before its runs.**

- **Runs:** the same 9 arms on the seed1 and seed2 models of the 8 cells (16 models, about 30 GPU-hours on astro).
- **Confirmatory analysis, over all 24 models:** R(k16) - R(k1) at d = 4 and at d = 8, pooled with equal cell weight. Hierarchical bootstrap (models within cell, then episodes; 10,000 resamples). One-sided, alpha = 0.05 with Holm's correction over the two delays.
- **d = 2** is reported as secondary: the pilot showed no benefit at the shortest delay.
- **Fewer anchors:** K = 4 vs K = 1 is secondary too. The pilot shows K = 4 matching K = 16 on several cells.

**Batch-1 latency of hedging** (`latency/timing_hedge.py`, L40S, not exclusive, job 24947339). Scoring all anchors in one batched call (`batch_anchors=True`, which matches the looped scores to 6e-6) makes hedging nearly free for one environment. RSSM walker:

| Configuration | Latency per call |
|---|---|
| K = 1, population 512 | 100-105 ms |
| K = 4, population 512 | 102-106 ms |
| K = 16, population 128 | 103-109 ms |
| K = 16, population 512 | 156-161 ms |

Single-environment latency is set by the H sequential model steps, not by the number of candidate rows. The sequential first implementation cost K times the latency.

**Equal-latency comparison, written before its runs.** The arms whose single-environment latency equals plain committed-prefix planning (K = 1, population 512) within 3 % are:

- K = 4, population 512: already in the pilot and the replication.
- K = 16, population 128: new arms `hcpp{2,4,8}k16` with `--pop 128 --iters 6`, on all 24 models (job below).

Pilot (seed0) for K = 4 vs K = 1, both at population 512: d = 2: -0.004 [-0.023, +0.016]; d = 4: +0.017 [-0.010, +0.044]; d = 8: +0.037 [+0.013, +0.064].

Confirmatory, over 24 models with the hierarchical bootstrap: R(K=4, pop 512) - R(K=1, pop 512) and R(K=16, pop 128) - R(K=1, pop 512) at d = 4 and d = 8. One-sided, Holm over the four contrasts.

**Diffusion hedging (approved 5 October 2026; written before its runs).**

Setup:
- Model: the 32k seed0 diffusion snapshot, 2 sampler steps, population 64, 2 iterations, H = 32 decisions.
- Tasks: cartpole and cup catch. Walker is at the floor.
- Arms: `hcpp{8,16}k{1,4}` (`latency/run_diffusion.py`). The committed prefix is sampled once per episode. Each anchor is the context window (last 4 latent frames) with the predicted prefix frames plus one residual. The residual bank holds 128 d-step residuals of the sampler on D_train episodes 0-99.
- K = 1 is committed-prefix planning in its roll-once form. It differs from the earlier `cpp` arm, which samples the prefix afresh for every candidate.
- Episodes: 10 per arm, as two jobs of 5 (seeds 950,000-004 and 950,005-009). All arms run on astro (L40 / L40S).
- Budget: about 43 GPU-hours (K = 4 is about 4x K = 1).

Analysis, descriptive as for the other diffusion arms (no pooled test):

- R(K4) - R(K1) per task and delay, with episode-level paired standard errors.
- Reported alongside paused(d), aligned(d) and cpp(d) from Study 3.

Smoke test passed (job 24955374).

**Diffusion hedging result (5 October 2026, 23:30; `results/latency/diffusion_hedge_summary.json`).** All 16 jobs completed. R(K4) - R(K1), with episodes paired (10 per arm):

| Task | d | R(K4) - R(K1) |
|---|---|---|
| Cup catch | 8 | +0.177 ± 0.094 |
| Cartpole | 8 | +0.011 ± 0.031 |
| Cup catch | 16 | +0.084 ± 0.084 |
| Cartpole | 16 | -0.053 ± 0.021 |

Inconclusive at this sample size. The paper reports it descriptively as a feasibility demonstration (Table 4).

The exclusive-node timing is complete on V100, L40S and H200. The default planner's L95 for one environment is 126-161 ms on the V100, 77-98 ms on the L40S and 103-137 ms on the H200.

---

## 13. The proposed method: delay-error model and hand-off-aware planning (6 October 2026, written before its runs)

**Why.** CLIS-MPPI (CDC 2026) already hedges MPPI over sampled execution-time states under delay; see the novelty-audit addendum of 6 October. A second prior-art search found close precedent for each component, and no work combining them for learned world models:

- FutureRTC and Decoupled Delay Compensation correct the post-delay state with a learned residual.
- SLS² fits a heteroscedastic latent-error model for robust MPC, without delay.
- Filter-Aware MPC and Robust Predictable Control penalise uncertainty during planning.

The proposed method (`latency/delay_error.py`) is:

1. A post-hoc delay-error model, fitted on the frozen world model's own d-step residuals on D_train. It is conditioned on the predicted latent, the current latent and the committed actions, and predicts the residual's mean and variance.
2. Hedging over N(z_hat + mean, var), with K = 4 anchors. These are batched, so they add no latency at batch size 1.
3. A hand-off penalty. A second head predicts the log-variance of the next call's anchor from (state, actions to be committed). Candidates are penalised by lam x std(J) x the z-score of that prediction, so the planner prefers commitments whose outcome it can predict.

**Feasibility (offline, `latency/error_model.py`, 5 models, d = 4 and 8, `results/latency/error_model/`).**

- The mean correction removes 2-38 % of the anchor error.
- The 95 % intervals cover 85-94 % of the errors.
- Hedging with the learned variance is the best or tied-best of the four options in 5 of the 6 cases where there is regret to remove. The unconditional bank sometimes hurts.

**Tuning.**

- Models and episodes: the seed0 models (8 cells), 20 episodes from seeds 970,000-970,019, disjoint from every other run.
- Delays and arms: d in {4, 8}; `ecpp{d}k4`, `ecpp{d}k4h{0.25, 0.5, 1.0}` and `ecpp{d}k4r{0.25, 0.5, 1.0}`.
- Rule (`latency/select_lambda.py`): lam_h and lam_r are each the argmax of pooled normalised return over 8 cells x 2 delays. It is applied automatically by a dependent job, which then launches the confirmatory array.

**Confirmatory.**

- Models and episodes: the seed1 and seed2 models (16, never used for tuning), seeds 950,000-950,049, 50 episodes, d in {4, 8}, population 512, 6 iterations.
- New arms: `ecpp{d}k4`, `ecpp{d}k4h{lam_h}`, `ecpp{d}k4r{lam_r}`, `mcpp{d}` (mean correction only) and `scpp{d}k4` (homoscedastic hedge). Existing arms from the replication: `hcpp{d}k4` and `hcpp{d}k1`.
- Contrasts, each at d = 4 and 8, one-sided, Holm over all eight, hierarchical bootstrap over 16 models in 8 cells:
  1. ecpp k4 vs hcpp k4: learned vs unconditional hedge.
  2. ecpp k4 h vs ecpp k4: the hand-off penalty adds return.
  3. ecpp k4 h vs ecpp k4 r: the hand-off penalty beats a generic mean-std risk penalty of the same form.
  4. ecpp k4 h vs hcpp k1: the full method vs plain committed-prefix planning.
- Secondary: mcpp and scpp, against ecpp k4 and hcpp k1.

**Budget.** About 15 GPU-hours on astro (tuning about 5, confirmatory about 10), roughly 3-4 hours of wall-clock.

**Section 13 result (6 October 2026, 13:35; `results/latency/de_confirm.json`, `results/latency/tune_selection.json`).**

**Tuning:**

- The hand-off penalty lowered return at every strength. Pooled tuning return was 0.313 without it, and 0.298, 0.251 and 0.227 at lam 0.25, 0.5 and 1.0.
- The risk penalty peaked at lam 0.5 (0.330).
- Chosen: lam_h = 0.25, lam_r = 0.5.

**Primary contrasts, held out:** none supported.

| Contrast | d = 4 | d = 8 |
|---|---|---|
| C1: learned vs bank | +0.001 | +0.011 |
| C2: hand-off vs none | -0.026 | -0.021 |
| C3: hand-off vs risk | -0.011 | +0.006 |
| C4: full vs committed prefix | -0.001 | +0.026 |

**Secondary:**

| Contrast | d = 4 | d = 8 |
|---|---|---|
| Bank vs committed prefix | +0.025 | +0.037 |
| Learned vs committed prefix | +0.025 | +0.047 |
| Learned vs time-aligned | +0.044 | +0.090 |
| Mean correction only vs committed prefix | +0.015 (n.s.) | +0.020 (n.s.) |
| Fixed vs learned spread | +0.008 | -0.020 |

**Decision (the author's choice, option A):** the paper's method is residual-calibrated hedging (RCH): the residual bank, with the learned error model as an optional variant. The hand-off penalty and the risk penalty are reported as failed ablations. The manuscript was rewritten as v4 on 6 October; v3 is in `archive/main_v3_before_rch.tex`.

## 14. Ensemble and particle hedging baselines (11 October 2026, written before the runs)

**Why.** Reviewer feedback on the manuscript: it discusses DATS and CLIS-MPPI but does not compare against a DATS-style ensemble-hedging baseline. The novelty audit also left this comparison as an outstanding check. Requested: committed-prefix planning without hedging, RCH with the model's calibration residuals, and an ensemble- or particle-based hedge. They should use the same environments and delays and be compared at comparable latency and, where feasible, comparable candidate population. The ensemble's extra model, memory and compute requirements should be reported.

**Baselines (code: `latency/ensemble.py`).**

- *Ensemble hedging, DATS-style.* The ensemble is the M = 3 independently trained models of each family-task cell (seeds 0, 1, 2), the same checkpoints as every other study. DATS propagates ensemble particles through the committed actions and averages each candidate's cost over them. Pixel world models trained from different seeds have incompatible latent spaces, so a particle cannot switch members between steps. Each member therefore keeps one particle for the whole rollout (TS-inf propagation):
  - every member filters its own latent from the same frames at every decision;
  - at each call, member m rolls the committed prefix through its own dynamics and scores the shared candidates' free part with its own reward head;
  - a candidate's cost is the mean over members.
- *Particle hedging, RSSM only* (`pcpp<d>k<K>`, single model, `latency/run_dmc.py`). Anchor 1 is the prediction, as in RCH. Anchors 2..K are sampled from the RSSM's own stochastic transition through the committed actions, instead of being drawn from the residual bank. Only the source of the hedge states differs from RCH. JEPA and TD-MPC2 are deterministic, so this baseline does not apply to them.

**Arms.** All at d in {4, 8}, 50 episodes from seeds 950,000-950,049, planner noise seed 777: the same initial states and planner noise stream as every other arm. Eligible cells: the 8 cells of Study 1.

| Arm | Models | Population | Iterations | Rollouts per candidate per iteration |
|---|---|---|---|---|
| CPP: `hcpp{d}k1` (existing) | 24 single models | 512 | 6 | 1 |
| RCH: `hcpp{d}k4` (existing) | 24 single models | 512 | 6 | 4 (batched) |
| E-pop: `ens{d}` | 8 ensembles of 3 | 512 | 6 | 3 (one per member) |
| E-lat: `ens{d}` | 8 ensembles of 3 | 512 | 2 | 3 |
| E+RCH: `ens{d}k4` (exploratory) | 8 ensembles of 3 | 512 | 6 | 12 |
| Particles: `pcpp{d}k4` | 9 RSSM models | 512 | 6 | 4 |

- **E-pop** matches the candidate population and the iterations of CPP and RCH.
- **E-lat** matches their batch-1 latency when the members are evaluated one after another. Each call then makes 3 x 2 = 6 sequential rollouts, the same as one model at 6 iterations. `latency/timing_ensemble.py` checks this.
- If a fused implementation evaluated the members in one batched call, the ensemble's latency could approach that of one model. E-pop is then also the latency-matched comparison. Both readings are reported.

**Statistics (`latency/analyse_ensemble.py`).**

- Each cell gives paired differences: the ensemble's normalised return on an episode minus each single model's return on the same initial state, a 3 x 50 array.
- Cells are weighted equally.
- Intervals come from a hierarchical bootstrap (models within cell, then episodes), with 10,000 resamples.
- The ensemble is one model set per cell, so its intervals do not include variation between ensembles. This is stated as a limitation.
- Primary contrasts, two-sided because no direction is claimed, with Holm over the four:
  1. RCH vs E-pop at d = 4 and 8;
  2. RCH vs E-lat at d = 4 and 8.
- Secondary contrasts, two-sided:
  - E-pop vs CPP and E-lat vs CPP;
  - E+RCH vs E-pop and E+RCH vs RCH;
  - particles vs RCH and particles vs CPP, on the 9 RSSM models.

**Interpretation, fixed before the runs.**

- RCH vs E-lat interval includes zero or favours RCH: at matched latency and without extra models, RCH matches or exceeds ensemble hedging.
- E-pop beats RCH: the paper reports that a three-model ensemble gains more at matched population, at three times the models, training, weight memory and per-call model evaluations. RCH is then positioned as the single-model option, and E+RCH shows whether the two combine.
- RCH vs E-pop interval includes zero or favours RCH: RCH matches a three-model ensemble with one model.
- Every outcome is reported in the paper.

**Resource accounting (`latency/timing_ensemble.py`, three cells: RSSM walker, JEPA cup, TD-MPC2 cup).**

- Parameters and GPU weight memory per member.
- Peak planning memory above the weights at batch size 1.
- Batch-1 call latency (L50 and L95) of CPP, RCH, E-pop, E-lat, E+RCH and particles.
- The per-decision cost of filtering one latent against three.
- The training cost of a member: from the MetricTrust training jobs, 0.8 (TD-MPC2), 1.3 (JEPA) and 2.2 (RSSM) V100 GPU-hours per model.

**Budget.** About 8 GPU-hours on astro: ensemble arms about 6, particles about 1.5, timing about 0.5. Wall-clock about 2-3 hours at the per-user cap.

**Launch.**

- Smoke test: job 25104146.
- Arrays: `sbatch --array=0-15 --export=ALL,LINES=latency/runs_ensemble.txt slurm/latency_ens.sh`; `sbatch --array=0-8 --export=ALL,LINES=latency/runs_particle.txt slurm/latency_dmc.sh`.
- Timing: `sbatch slurm/latency_timing_ens.sh`.

**Section 14 primary result (11 October 2026; `results/latency/ensemble.json`).**

| Contrast | d = 4 | d = 8 |
|---|---|---|
| RCH vs E-pop | -0.023 [-0.051, +0.002], Holm p = 0.21 | -0.008 [-0.038, +0.021], Holm p = 0.60 |
| RCH vs E-lat | -0.042 [-0.070, -0.016], Holm p = 0.0008 | -0.026 [-0.057, +0.004], Holm p = 0.21 |

- Under the pre-specified reading, the three-model ensemble beats RCH at matched latency at d = 4.
- Against the ensemble at matched population, RCH is lower at both delays, but neither interval excludes zero.
- Secondary:
  - E-pop vs CPP: +0.046 and +0.045.
  - E-lat vs CPP: +0.064 and +0.063.
  - E+RCH vs E-pop: -0.003 and +0.033.
  - E+RCH vs RCH: +0.021 and +0.041.
  - Particles vs RCH (RSSM): +0.020 and -0.000.

**Post-hoc check, written after the primary result and before its runs.** E-lat (2 iterations) scored above E-pop (6 iterations), so iteration count may confound the latency-matched contrast.

- In Study 2, the seed-0 models show no consistent effect for single-model committed-prefix planning: 0.366 vs 0.367 at d = 4 and 0.284 vs 0.275 at d = 8, for 2 vs 6 iterations.
- To settle it, `hcpp{d}k1` and `hcpp{d}k4` are run at population 512 and 2 iterations on the 24 models, for d in {4, 8} (`latency/runs_hedge_it2.txt`).
- This is reported as post-hoc, with no correction. Contrasts:
  - RCH(it2) vs E-lat;
  - RCH(it2) vs RCH(it6);
  - CPP(it2) vs CPP(it6).

**Second post-hoc check, written after the primary result and before its runs.** Single models vary widely within a cell; for example, the three JEPA cup models score 0.61, 0.73 and 0.37 at d = 4. The ensemble roughly reaches the best member, so part of its gain may be better modelling rather than better delay compensation.

- Arms: `enspaused{1,4,8}`, the ensemble with the simulator paused while it plans (8 cells; population 512; 6 and 2 iterations; `latency/runs_ensemble_paused.txt`).
- Reported quantities:
  - the ensemble's gain without delay, E-paused1 vs the single models' paused1;
  - the ensemble's own feedback-rate and mismatch losses, F_E(d) = R_E(paused1) - R_E(paused_d) and M_E(d) = R_E(paused_d) - R_E(ens_d);
  - the share of each planner's mismatch loss that it removes.
- Post-hoc, with no correction.

**Post-hoc results (11 October 2026; `results/latency/ensemble.json`, sections `posthoc` and `decomposition`).**

*Iteration count* does not explain the latency-matched result:

| Contrast | d = 4 | d = 8 |
|---|---|---|
| RCH at 2 vs 6 iterations | -0.003 [-0.019, +0.012] | -0.005 [-0.024, +0.013] |
| CPP at 2 vs 6 iterations | +0.008 [-0.008, +0.025] | +0.004 [-0.016, +0.024] |
| RCH (2 iterations) vs E-lat | -0.045 [-0.070, -0.022] | -0.031 [-0.055, -0.006] |

*Decomposition.* The ensemble is better without delay: E-paused1 vs the single models' paused1 is +0.066 [+0.033, +0.099]. Its gain under delay (+0.046, +0.045 over CPP) is no larger.

Mismatch loss, R(paused_d) - R(planner_d), each against its own paused planner:

| Planner | d = 4 | d = 8 |
|---|---|---|
| CPP | 0.076 | 0.098 |
| RCH | 0.054 | 0.061 |
| E-pop | 0.076 | 0.111 |
| E-lat | 0.068 | 0.103 |
| E+RCH | 0.079 | 0.078 |

Differences in mismatch loss:

| Contrast | d = 4 | d = 8 |
|---|---|---|
| M_CPP - M_E-pop | -0.000 [-0.023, +0.023] | -0.012 [-0.036, +0.011] |
| M_RCH - M_E-pop | -0.022 [-0.049, +0.004] | -0.049 [-0.075, -0.024] |
| M_RCH - M_E-lat | -0.015 [-0.040, +0.010] | -0.041 [-0.067, -0.015] |

- The ensemble's lead is model quality (averaging three models), present with or without delay. Its committed-prefix planner loses as much to mismatch as a single model's.
- RCH is the planner that reduces mismatch loss. On the ensemble, it does so at d = 8 (E+RCH).

**Timing (job 25105777, L40S, round-robin; `results/latency/timing_ensemble/l40s/`).**

- Call latency at d = 4:
  - one model, CPP or RCH: 83-103 ms;
  - ensemble at 6 iterations: 239-299 ms (2.9x);
  - ensemble at 2 iterations: 80-100 ms;
  - ensemble + RCH: 242-303 ms;
  - particles (RSSM): 107 ms.
- Peak planning memory above the weights: 10-18 MiB for one model or the ensemble, 28-46 MiB with four anchors.
- Weights per member: 13-35 MiB.
- Filtering per decision: 0.46-0.77 ms for one model, 1.33-2.23 ms for three.
- The first timing run (job 25104721) timed the configurations one after another and showed drift (K = 1 slower than K = 4); its files are kept in scratch, `runs/timing_ensemble_v1`.

**Compute used:** 13.2 GPU-hours. Ensemble 4.7, particles 1.5, RCH at 2 iterations 2.2, paused ensemble 4.3, timing and smoke test 0.4.

**Manuscript (11 October 2026):**

- New Results subsection, "An ensemble improves the model, and residual hedging the handling of delay", with Tables 7 and 8.
- Methods subsection "Ensemble and particle baselines"; pre-registration entry; statistics, compute and a recipe step.
- Abstract, Introduction and Discussion revised.

## 15. Studies for the PRX Intelligence submission (16 October 2026, written before their runs)

**Why.** A review of the manuscript judged the contribution incremental:
- modest gains at longer delays and none at short delays;
- outperformed by a three-model ensemble;
- many unsupported pre-registered hypotheses;
- a straightforward extension of existing hedging.

The validation (16 October) confirmed these points. It also found that the K = 4 equal-latency contrast was designated confirmatory after the seed-0 pilot was seen.

The three studies below test what the claim of a general, adoptable method needs and the current evidence lacks:
- the delay range where mismatch is largest;
- a third-party pretrained world model, where an ensemble is not available;
- real-time execution.

Each has fresh data and a confirmatory contrast fixed here.

### 15A. Longer delay on the 24 models

- **Arms:** `hcpp16k1` and `hcpp16k4`, population 512, 6 iterations, H = 32, 50 episodes from seeds 950,000-950,049, on all 24 models. `paused16` and `aligned16` already exist from Study 1.
- **Confirmatory:** R(K=4) - R(K=1) at d = 16, pooled with equal cell weight, with the hierarchical bootstrap. One-sided, alpha = 0.05; a single test.
- **Secondary:** the mismatch loss of each arm against `paused16`.
- **Budget:** about 3 GPU-hours on astro.

### 15B. A third-party pretrained world model: LeWM

- **Models:** the public LeWM checkpoints (Maes et al. 2026; `quentinll/lewm-{pusht,cube,reacher}` on the Hugging Face Hub). These are JEPA world models of about 15M parameters trained by another group, with one released checkpoint per task. TwoRoom is excluded by the overlap rules with the main paper.
- **Planner:** LeWM's own CEM settings (300 samples, 30 iterations, top 30, horizon 5 model steps of 5 environment steps each), with one change: it replans every model step instead of every 5. A decision is one model step.
- **Goal and cost:** the goal image and LeWM's latent goal cost, unchanged.
- **Delay schedules:** as in `latency/schedule.py`, at d in {1, 2} decisions. One planning call of LeWM takes about 1 s, so d of 1-2 decisions is the delay these planners incur in real time. `latency/lewm.py` measures the batch-1 latency.
- **Arms:** paused1, paused_d, aligned_d, cpp_d and rch_d (K = 4).
  - The residual bank holds 512 windows of the LeWM training dataset. Each residual is r = enc(o_{t+d}) - rollout(z_t, a_{t:t+d}).
  - RCH replaces the last embedding of the predictor's history after the prefix with each anchor. Anchors are scored in one batched call.
- **Episodes:** 200 start-goal pairs per task, sampled from LeWM's evaluation dataset with seed 950,000 (LeWM's own evaluation uses seed 42) and shared across arms. LeWM's evaluation protocol is otherwise unchanged: goal offset 25 steps, budget 50 steps, LeWM's success criterion.
- **Confirmatory:** success(rch_d) - success(cpp_d), pooled over the three tasks with equal weight, at d = 1 and d = 2. One-sided, Holm over the two delays, paired bootstrap over episodes (10,000 resamples).
- **Secondary:**
  - cpp_d vs aligned_d;
  - mismatch loss against paused_d;
  - per-task results.
- **Implementation details** that the harness forces are logged as amendments before the runs. Planner settings and contrasts do not change.

### 15C. Real-time hedging

- **Setup:** `latency/wallclock.py` with a new arm, `rch` (K = 4, anchors batched), on the six cup-catch models of JEPA and TD-MPC2.
  - These are the two cells where the fixed-delay study found the largest gains, so this tests whether the gain survives real-time execution. It does not test generality.
  - V100, node held exclusively, population 512, 6 iterations, 30 episodes per arm from seeds 950,000-950,029.
  - Arms: cpp, rch and aligned.
- **Delay:** the earlier real-time check under-estimated the loop delay, so d_est is now calibrated from one unscored real-time episode. It is the 95th percentile, rounded up, of the realised decisions per call, with the slowest arm (rch).
- **Confirmatory:** return(rch) - return(cpp), pooled over the six models with equal cell weight, using the hierarchical bootstrap. One-sided, alpha = 0.05.
- **Secondary:** cpp vs aligned; realised call durations and underruns.

### Text corrections, independent of these results

- State that the K = 4 equal-latency contrast became confirmatory after the seed-0 pilot. Report its estimate on seeds 1 and 2 alone: +0.025 [+0.003, +0.047] and +0.037 [+0.011, +0.064].
- Correct "little mismatch remains" at d = 2. The mismatch is 0.042 there; hedging does not convert it into return.
- State that the gains concentrate in the cup-catch cells and RSSM walker.

**Budget:**
- 15A: about 3 GPU-hours.
- 15B: about 20-40 GPU-hours on astro after the harness is built, with 77 GiB of public data.
- 15C: about 4 hours of an exclusive V100.

### Amendment to 15B (16 October 2026, after the harness checks and before any confirmatory run)

**Harness checks** (not results):
- `latency/lewm.py` reproduces LeWM's latent cost to 9e-5 (cost scale 270).
- Committed-prefix planning equals the cost of the concatenated sequence, and RCH with zero anchors equals committed-prefix planning, both to the same tolerance.
- On LeWM's own protocol (`latency/lewm_validate.py`; seed 42, 50 PushT episodes, goal offset 25, budget 50), LeWM's reference policy succeeded 0.92. LeWM reports about 0.90. Our policy replanning every 5 decisions succeeded 0.98.

**Why the design changes.** Replanning every decision succeeded only 0.32. LeWM's cost scores only the final predicted embedding, 5 decisions ahead, so replanning every decision keeps deferring arrival at the goal. The pre-registered change ("replans every model step") is therefore withdrawn. A 10-episode smoke test had run the withdrawn design; it is not used.

**Amended design.** LeWM's planner and receding horizon are kept: each plan's 5 blocks are executed before the next plan takes over.
- paused5: the simulator waits for each call. This is LeWM's own evaluation.
- Asynchronous schedules: a call takes d decisions, so the next call is issued d decisions before the current plan runs out.
  - cpp_d and rch_d take the d remaining blocks as the committed prefix, and the first 5 free blocks are executed.
  - aligned_d plans d + 5 blocks from the observation and executes blocks d:d+5.
  - naive_d executes blocks 0:5.
- Every schedule therefore replans every 5 decisions. paused5 minus an asynchronous schedule is pure mismatch loss, with no feedback-rate loss.
- With 5 blocks per plan and calls of d decisions, the latency schedules of `latency/schedule.py` are the case where both equal d.

**Delays: d in {2, 4} decisions.** LeWM's call takes about 1 s (0.98 s reported), so these are the delays of PushT, at 0.5 s per decision, and OGBench Cube, at about 0.25 s per decision. Reacher's decision is 0.1 s, so its delay of about 10 decisions exceeds a plan; it is run at the same two delays.

**Settings, 200 episodes per task and setting, seed 950,000:**
- **S2, long horizon (confirmatory):** goal offset 50, budget 100; three asynchronous calls per episode.
- **S1, LeWM's protocol (secondary):** goal offset 25, budget 50; one asynchronous call per episode, and paused success of 0.92-0.98 leaves little room.

**Confirmatory:** success(rch_d) - success(cpp_d) in S2, pooled over the three tasks with equal weight, at d = 2 and d = 4. One-sided, Holm over the two delays, paired bootstrap over episodes within task (10,000 resamples).

**Secondary:**
- The same contrast in S1.
- cpp_d vs aligned_d.
- Mismatch loss, paused5 - X_d.
- Per-task results.
- Batch-1 call latency.

**Arms per task and setting:** paused5, naive{2,4}, aligned{2,4}, cpp{2,4} and rch{2,4}k4.

**Analysis:** `latency/analyse_lewm.py`.

**Budget:** about 10-20 GPU-hours on astro.

### Amendment 2 to 15B (16 October 2026, after smoke tests and before any confirmatory run)

**S2 is dropped.** In S2 (goal offset 50, budget 100; 20 episodes per arm), delay did not change success:
- PushT: paused5 0.20, every asynchronous arm 0.20-0.25.
- Cube: paused5 0.45, every asynchronous arm 0.40-0.45.

LeWM rarely reaches a goal 50 steps away, and the first plan, made at rest, is never delayed. S1 has the same limitation: one delayed call per episode, after a first plan that succeeds about 92% of the time.

**Hand-over setting (H, confirmatory).** Delay compensation matters when a plan must be computed while the system moves. In H, each evaluation episode starts with the system executing the expert's recorded actions from the start state (the dataset's actions; `latency/lewm.py --handover d`). The planner's first call is issued at the start and lasts d decisions:
- cpp_d and rch_d take the d recorded blocks as the committed prefix;
- aligned_d and naive_d plan d + 5 blocks from the start observation;
- paused5 executes the d recorded blocks and then plans from the true state with the simulator paused, every 5 decisions. This is the exact-model counterpart of cpp_d.

Later calls follow the amended schedule: each is issued d decisions before the current plan runs out. The goal lies 25 steps beyond the hand-over, as in LeWM's protocol: goal offset 25 + 5d (35 and 45) and budget 5d + 50 (60 and 70).

The setting was chosen for its delay sensitivity, judged by paused5 against naive and aligned. In a 30-episode PushT smoke test at d = 2, paused5 succeeded 0.90, naive 0.07 and aligned 0.30. Smoke results for cpp and rch were also seen (30 episodes; differences in both directions) and are not used.

**Seed 951,000.** The confirmatory episodes use seed 951,000 instead of 950,000. The 200 episodes per task share no episode with the smoke tests; seed 950,000 would have shared one row per setting.

**Confirmatory (replaces the 15B and amendment 1 contrasts).** success(rch_d k4) - success(cpp_d) in H, pooled over pusht, cube and reacher with equal weight, at d = 2 and d = 4. One-sided, Holm over the two delays, paired bootstrap over episodes within task (10,000 resamples), 200 episodes per task and arm.

**Secondary:**
- The same contrast in S1 (LeWM's protocol with amendment 1's schedules; seed 951,000).
- cpp_d vs aligned_d.
- Mismatch loss, paused5 - X_d.
- Per-task results.
- Batch-1 call latency.

**Runs.** `latency/runs_lewm.txt`: six hand-over jobs (task x d) and three S1 jobs.

### Section 15 results (16 October 2026)

**15A, d = 16, 24 models (`results/latency/d16.json`): supported.**
- R(K=4) - R(K=1) = +0.023 [+0.004, +0.042], one-sided p = 0.0094.
- Mismatch loss falls from 0.084 to 0.061.
- The gain is concentrated: JEPA cup +0.147, TD-MPC2 cup +0.030, RSSM walker +0.012; the other cells change by less than 0.006.

**15C, real time, 6 cup-catch models (`results/latency/wallclock_v2.json`): not supported.**
- rch - cpp = +0.074 [-0.029, +0.181], one-sided p = 0.082.
- By model: JEPA +0.107, +0.039, +0.306; TD-MPC2 -0.037, -0.101, +0.129.
- Secondary: cpp - aligned +0.118 [+0.034, +0.205]; rch - aligned +0.192 [+0.102, +0.284].
- Calibrated d: 6 for JEPA, 5 for TD-MPC2, one decision above the start-up estimate. There were no underruns, and no call exceeded d_est by more than one decision.

**15B, LeWM, hand-over setting, 200 episodes per task and arm, seed 951,000 (`results/latency/lewm_summary.json`): not supported.**

| | d = 2 | d = 4 |
|---|---|---|
| rch - cpp (Holm) | +0.002 [-0.027, +0.030], p = 0.48 | +0.015 [-0.013, +0.043], p = 0.34 |
| Per task (pusht / cube / reacher) | +0.005 / -0.055 / +0.055 | -0.010 / -0.010 / +0.065 |
| cpp - aligned | +0.303 [+0.262, +0.345] | +0.333 [+0.292, +0.375] |

- Success by arm: PushT d = 2: paused5 0.90, naive 0.08, aligned 0.31, cpp 0.835, rch 0.84.
- Mismatch loss paused5 - cpp:
  - PushT: 0.065 and 0.035;
  - Cube: 0.010 and -0.015;
  - Reacher: -0.23 and -0.25. Committed-prefix planning beats the paused reference there. The likely cause is that the prefix rollout gives LeWM's predictor three frames of context, while the paused planner plans from one frame (LeWM's history length 1), so the paused reference is not an exact-state reference for LeWM.
- S1 (LeWM's protocol): every asynchronous arm is within 0.03 of paused5; rch - cpp = +0.002 at both delays.

**LeWM batch-1 latency (L40S, PushT; first timing run, job 25235800):**
- paused and cpp: 339-345 ms;
- rch with K = 4: 1148-1162 ms.

Unlike the smaller latent models, LeWM's transformer predictor makes four anchors cost 3.4x the call time. The cube and reacher timing failed (hard-coded action dimension) and are rerun as job 25243321.

**Reading (written after the results).**
- RCH's gains are confined to models and tasks with a large value-aligned mismatch: the JEPA and TD-MPC2 cup-catch cells and RSSM walker.
- On a third-party JEPA world model, committed-prefix planning leaves little mismatch, and hedging adds nothing.
- In real time the estimate is positive but not significant.

## 16. Second round for PRX Intelligence (16 October 2026, written before the runs)

**Why.** After Section 15, RCH's evidence has three gaps:
- its gains are concentrated in two cup-catch cells and RSSM walker, from only three tasks;
- the real-time test was underpowered (+0.074, p = 0.082, 30 episodes per arm);
- the third-party model left too little mismatch to test the method.

The studies below address the first two directly. The third gap is addressed indirectly, with a pre-specified test of whether an offline statistic predicts where hedging helps.

### 16A. Powered real-time replication

- **Setup:** the protocol of 15C unchanged (six cup-catch models; calibrated d_est; arms rch, cpp and aligned; population 512, 6 iterations; exclusive V100). It uses 100 new episodes per arm and model, initial-state seeds 950,100-950,199, disjoint from 15C and from the calibration seed.
- **Confirmatory:** real-time return(rch) - return(cpp) on these new episodes, normalised, pooled with equal cell weight, hierarchical bootstrap. One-sided, alpha = 0.05.
- **Secondary:** the same pooled with the 15C episodes (130 per arm and model); cpp vs aligned.
- **Code:** `latency/wallclock.py --seed0 950100 --episodes 100 --calibrate`, results in `results/latency/wallclock_v3`.

### 16B. Three new tasks

- **Tasks:** the other three tasks of the PlaNet pixel benchmark, reacher-easy, cheetah-run and finger-spin. Together with cartpole swing-up, walker walk and cup catch they make up the whole benchmark. They are chosen for that reason, not for any expected effect.
- **Pipeline:** the main study's, unchanged except for the tasks.
  - 2,000 episodes per task, collected with an epsilon ladder over the released TD-MPC2 agent at 128 px and packed to 64 px; episodes 1,900-1,999 are held out.
  - RSSM, JEPA (3-frame history) and TD-MPC2, with 3 seeds each and the same training steps (60k, 60k, 100k): 27 new models. They are trained only for this paper and enter no analysis of the main study.
- **Normalisation:** (R - R_rand) / (R_exp - R_rand), where R_exp is the mean return of the epsilon = 0 tier of the collected data and R_rand that of the epsilon = 1 tier.
- **Floor rule:** as in Study 1, a cell enters if its mean paused1 return over the three models is at least 0.15.
- **Arms** (population 512, 6 iterations, H = 32, 50 episodes from seeds 950,000-950,049): paused1, paused4, paused8, aligned4, aligned8, hcpp4k1, hcpp4k4, hcpp8k1 and hcpp8k4.
- **Confirmatory:** R(K=4) - R(K=1) at d = 4 and d = 8, pooled with equal cell weight over the eligible new cells, hierarchical bootstrap. One-sided, Holm over the two delays.
- **Secondary:**
  - committed prefix vs time-aligned;
  - the mismatch loss and the share of it removed by committed prefix and by RCH;
  - per-cell gains;
  - the same contrasts pooled over all 6 tasks.

### 16C. When to hedge (secondary, pre-specified)

- **Statistic:** for each model and d in {4, 8}, the offline hedging value V = (regret of the committed-prefix plan - regret of the K = 4 hedged plan) / regret of the stale-latent plan.
  - It uses the diagnostic of `latency/diagnose_anchor.py`, on 64 held-out windows with the recorded actions as the prefix.
  - Each regret is the model's cost from the true post-delay latent, relative to the plan made from that latent.
- **Prediction:** the cell-mean V and the cell-mean closed-loop gain R(K=4) - R(K=1) are positively rank-correlated (Spearman, one-sided) across the (cell, d) pairs of all eligible cells.
  - For the three original tasks this is a retrodiction.
  - For the new tasks V is computed before their closed-loop results are read.
- **Exploratory:** the same for LeWM, whose results are known.

### Budget

- 16A: about 6 x 6 hours of exclusive V100.
- 16B: collection about 5 GPU-hours, training about 40, closed loop about 15.
- 16C: about 5.
- Wall-clock about one to two days.

### Section 16 results so far (16 October 2026)

**16A, powered real-time replication (`results/latency/wallclock_v3.json`): not supported at alpha = 0.05.**
- rch - cpp = +0.071 [-0.013, +0.150], one-sided p = 0.051, on 100 new episodes per arm and model.
- The estimate agrees with 15C (+0.074). Pooled with the 15C episodes (secondary): +0.072 [-0.001, +0.138].
- By model: JEPA +0.231, -0.014, +0.186; TD-MPC2 +0.115, -0.021, -0.070.
- Secondary: rch - aligned +0.135 [+0.063, +0.211]; cpp - aligned +0.064 [+0.010, +0.117].
- Calibrated d: 6 for JEPA and 5 for TD-MPC2; no underruns.

**16C retrodiction on the original eligible cells (`results/latency/diag16.json`).**
- Spearman(V, closed-loop gain) = +0.69, one-sided permutation p = 0.002, over 16 (cell, d) pairs.
- Misses: TD-MPC2 cartpole at d = 4 (V 0.48, gain 0.00) and TD-MPC2 walker (V 0.10-0.14, gain 0.00).
- This is a retrodiction; the prediction for the new tasks is pending.

**16B pipeline.**
- Data collected and packed.
- Normalisation (eps 0 / eps 1 tiers): reacher-easy 972.9 / 57.1, cheetah-run 816.0 / 6.6, finger-spin 820.9 / 1.1.
- Agent choice by the pre-set rule: cheetah-run's expert tier uses MPC (prior 494 against MPC 850).
- Training, checkpoint copy, diagnostics and evaluation are chained (jobs 25243926-7, 25246742-4).

**16B deviation (16 October 2026, before training started).** All 24 V100s were occupied by the main study, so the 27 new models are trained on astro (L40/L40S) instead of V100. The trainers select bf16 autocast on these GPUs, where the original 24 models were trained on V100 with fp16. Everything else is unchanged. The LeWM offline hedging value (16C, exploratory) is V = -0.013 to +0.024 for every task and delay; see `results/latency/lewm/diag16.json`.

### Section 16 final results (16-17 October 2026)

**16B, three new tasks (`results/latency/newtasks.json`): supported.**
- Eligible new cells: RSSM, JEPA and TD-MPC2 on reacher-easy and finger-spin, plus RSSM cheetah-run. JEPA and TD-MPC2 cheetah-run fall below the 0.15 floor.
- R(K=4) - R(K=1): +0.025 [+0.010, +0.040] at d = 4 (Holm p = 0.0004) and +0.017 [+0.006, +0.027] at d = 8 (Holm p = 0.0018).
- Cells gaining: 5 of 7 at d = 4 and 6 of 7 at d = 8. The largest gain is RSSM cheetah's +0.073 at d = 4; no cell loses more than 0.010.
- Committed prefix vs time-aligned: +0.127 and +0.161.
- Mismatch loss: committed prefix 0.089 and 0.129; RCH 0.064 and 0.113.
- All six tasks pooled (15 cells, 45 models): +0.023 and +0.027, both p <= 0.0001.

**16C, when to hedge (`results/latency/diag16.json`).**
- Original cells (retrodiction): Spearman +0.69, p = 0.002.
- New cells (prediction): +0.33, p = 0.13. **Not supported.** RSSM cheetah has the largest gain with V < 0, and JEPA reacher gains with V < 0. The finger cells (V 0.06-0.23) all gain.
- All 30 pairs: +0.42, p = 0.010.
- LeWM (exploratory): V between -0.013 and +0.024, and no closed-loop gain.
- Reading: V tracks the gain where it is large (cup catch, finger), but it is not a reliable prospective rule.

## 17. Third round: a larger effect and a positive third-party test (17 October 2026, written before the runs)

**Why.** The remaining weaknesses:
- typical gains of 0.02-0.03;
- real time at p near 0.05;
- a null on a third-party model;
- a core idea close to existing hedging.

The stochastic-programming decomposition (Section 16C) says hedging recovers only the value of the stochastic solution. The larger remainder, the value of information, needs a fresher observation. 17A tests a method that attacks both parts at no extra compute or latency. 17B tests RCH on a widely used third-party model in the delay setting of DATS.

### 17A. Hedge-then-refine (re-anchored hedging)

**Method.** It keeps RCH's budget, 6 MPPI iterations per call, and splits them:
1. The first 5 iterations are RCH's, from the latent at the start of the call, with the d committed actions as the prefix and K = 4 anchors from the d-step residual bank.
2. The last iteration is re-anchored on the latent observed q = ceil(d/6) decisions before the plan takes effect. q is that iteration's share of the call's latency, rounded up. The last q committed actions are the prefix, and the iteration starts from the mean and standard deviation of the first 5. It hedges over K = 4 anchors from a q-step residual bank.

Total compute and latency equal RCH's, and the plan still takes effect d decisions after the call starts.

**Arms** (all 45 eligible models of the six tasks, population 512, 50 episodes, seeds 950,000-950,049, d in {4, 8}):
- `rcpp{d}k4r1`: hedge-then-refine;
- `rcpp{d}k1r1`: re-anchored committed prefix, no hedging.

Existing comparators are `hcpp{d}k1` and `hcpp{d}k4`.

**Confirmatory:** R(hedge-then-refine) - R(RCH, K = 4) at d = 4 and d = 8, pooled with equal cell weight over the 15 eligible cells, hierarchical bootstrap. One-sided, Holm over the two delays.

**Secondary:**
- hedge-then-refine vs committed prefix;
- re-anchored committed prefix vs committed prefix;
- hedge-then-refine vs re-anchored committed prefix;
- per-cell results and the share of the mismatch loss removed.

### 17B. Released TD-MPC2 world models (third party)

- **Models:** the official single-task, state-based, 5M-parameter TD-MPC2 checkpoints (Hansen et al.; seeds 1-3) for the six PlaNet tasks, 18 models. The planner is TD-MPC2's own MPPI: horizon 3, 6 iterations, 512 samples, 64 elites, 24 policy trajectories, value bootstrap. It is re-implemented in `envs/data_agent.py`; settings unchanged.
- **Delay setting:** pipelined constant delay, the setting of DATS.
  - A call starts at every decision from the current observation and returns d decisions later with one action, the one for decision t + d.
  - naive_d plans from the current latent as if no time passed.
  - cpp_d rolls the d pending actions through the model and plans from the predicted latent.
  - rch_d plans over K = 4 anchors, the prediction plus residuals from a d-step bank in TD-MPC2's latent space, built from the six tasks' collected datasets (D_train). Each candidate's value is the mean over anchors, and the policy-prior trajectories start from the prediction.
  - paused1, which replans every decision with the simulator waiting, is the reference. paused1 minus a delayed arm is pure mismatch loss.
  - The first d actions of every episode, for every delayed arm, come from TD-MPC2's policy prior rolled through its model from the first observation.
- **Delays:** d in {2, 4, 8}. 50 episodes per arm, seeds 950,000-950,049. Normalisation pairs as in the pixel study. Floor rule: paused1 of at least 0.15.
- **Confirmatory:** R(rch_d) - R(cpp_d) at d = 4 and d = 8, pooled with equal task weight over the eligible tasks (3 models each), hierarchical bootstrap. One-sided, Holm over the two delays.
- **Secondary:**
  - d = 2;
  - cpp vs naive;
  - mismatch loss paused1 - X_d;
  - per-task results;
  - the offline value V computed from TD-MPC2's own cost, as in Section 16C.

**Budget:** 17A about 15 GPU-hours; 17B about 10 GPU-hours plus implementation.

### Section 17 results (17 October 2026)

**17A, hedge-then-refine, 45 models in 15 cells (`results/latency/htr.json`): supported at d = 8, not at d = 4.**

| Contrast | d = 4 | d = 8 |
|---|---|---|
| HTR - RCH (confirmatory, Holm) | +0.008 [-0.004, +0.020], p = 0.08 | +0.015 [+0.004, +0.027], p = 0.0098 |
| HTR - CPP | +0.032 | +0.043 |
| RCH - CPP (for comparison) | +0.023 | +0.027 |
| Re-anchored CPP - CPP | +0.015 | +0.018 |
| HTR - re-anchored CPP | +0.017 | +0.025 |
| Mismatch loss, CPP -> HTR | 0.082 -> 0.050 (39 %) | 0.113 -> 0.070 (38 %) |

**17B, released TD-MPC2, 17 models in 6 tasks (`results/latency/tdmpc2_released.json`): not supported.**
- rch - cpp = +0.004 [-0.025, +0.033] at d = 4 and -0.019 [-0.078, +0.025] at d = 8 (Holm p = 0.81).
- cpp - naive: +0.66 to +0.73.
- cpp leaves a large mismatch: 0.018, 0.106 and 0.389 at d = 2, 4 and 8.
- Per task, hedging helped cheetah (+0.15 at d = 4) and hurt walker and finger.

**Diagnosis, after the result.** TD-MPC2's latent is 64 softmax groups of 8 (SimNorm). Adding another window's residual leaves the simplex: 6-16 % of anchor entries are negative, with negative mass per group of 0.01-0.11. The negative mass is largest on walker and finger, where hedging hurt, and smaller on cheetah and cup.

## 18. Follow-up: anchors in the latent's natural coordinates (17 October 2026, after 17B, before these runs)

**Hypothesis**, formed after the 17B result. For a softmax-group latent, residuals should be taken in log-probability coordinates, so that anchors stay on the model's latent manifold:
- per group g, rho_g = log z_g - log z_hat_g on the calibration window;
- the anchor is softmax_g(log z_hat_g + rho_g).

This is exact for the calibration window itself, and K = 1 is committed-prefix planning. The 17B null remains the pre-registered third-party result; this is a separate test.

### 18A (confirmatory)

- **Models:** the 17 released TD-MPC2 models.
- **Episodes:** new initial states, seeds 950,100-950,149, disjoint from 17B.
- **Arms:** paused1, cpp{4,8}, rch{4,8}k4 (additive, as in 17B) and rchl{4,8}k4 (log-space anchors). Pipelined delay, as in 17B.
- **Confirmatory:** R(rchl_d) - R(cpp_d) at d = 4 and d = 8, pooled with equal task weight, hierarchical bootstrap. One-sided, Holm over the two delays.
- **Secondary:** rchl vs rch.

### 18B (secondary)

- **Models:** our 15 pixel TD-MPC2 models in the 5 eligible cells (cartpole, cup, walker, reacher, finger), which use the same SimNorm latent.
- **Arm:** `hcpp{4,8}k4l` (log-space anchors), on the standard seeds 950,000-950,049, paired with the existing hcpp{d}k1 and hcpp{d}k4 arms.
- **Contrasts:** log vs additive anchors, and log anchors vs committed prefix. Reported without correction.

**Variants tried after 17B, reported in full:** this one.

### Section 18 results (17 October 2026)

**18A, log-probability anchors, released TD-MPC2, new episodes (`results/latency/s18.json`): not supported.**
- rchl - cpp = +0.008 [-0.020, +0.037] at d = 4 and -0.030 [-0.088, +0.017] at d = 8 (Holm p = 0.60 and 0.83).
- rchl - rch (additive anchors) = +0.008 and +0.004.
- The per-task pattern is unchanged: hedging helps cheetah and reacher and hurts walker and finger.
- The off-simplex explanation is rejected.

**18B, the 15 pixel TD-MPC2 models (secondary):**
- log - additive anchors: +0.008 and +0.001;
- log anchors - CPP: +0.028 (p = 0.013) and +0.016;
- additive anchors - CPP: +0.020 and +0.015.

**Manuscript:**
- New sections and tables: hedge-then-refine (Table XII) and the released TD-MPC2 models (Table XI), with the log-probability follow-up.
- The real-time iteration scheme is credited for re-anchoring.
- Abstract, Introduction, Discussion, Methods, pre-registration list and compute updated.

**Compute, Sections 17-18:** 22.5 GPU-hours.

## 19. Powered real-time test of hedge-then-refine (17 October 2026, written before the runs)

**Why.** Both real-time tests of RCH estimated about +0.07 but missed significance (p = 0.082 and 0.051). The intervals were wide because six models disagreed, not because of episode noise. This study tests the stronger planner, hedge-then-refine, on more models. Real time is where its equal-latency claim matters, since its refinement must fit inside the call.

**Cells.** The rule, fixed here from the fixed-delay results of 17A: cells whose hedge-then-refine gain over committed prefix, averaged over d = 4 and 8, is at least 0.03. Seven cells qualify: JEPA cup, TD-MPC2 cup, RSSM cheetah, JEPA finger, TD-MPC2 finger, TD-MPC2 reacher and RSSM walker, 21 models. Like 15C, this tests whether the gain survives real time on the cells where it exists. It is not a test of generality.

**Protocol (`latency/wallclock.py`).**
- Arms: htr, cpp and rch.
- htr runs rch's first five iterations from the latent at the call's start. It then filters the frames that arrived meanwhile and re-anchors the last iteration on the latest latent, with the committed actions still to run as its prefix. The refinement is skipped when none remain.
- d_est is calibrated from one unscored real-time episode of the slowest arm (htr), as in 15C.
- Population 512, 6 iterations, 50 episodes per arm and model, initial-state seeds 950,200-950,249 (new). V100 nodes held exclusively; results in `results/latency/wallclock_v4`.

**Confirmatory:** real-time normalised return(htr) - return(cpp), pooled with equal cell weight over the 7 cells, hierarchical bootstrap (models within cell, then episodes paired by seed). One-sided, alpha = 0.05.

**Secondary:**
- htr vs rch, and rch vs cpp;
- realised decisions per call, underruns, and the share of calls refined.

**Budget:** 21 jobs of about 1 hour each on exclusive V100 nodes.

# Novelty audit of the side paper (5 October 2026)

**Question.** Are the four claims of *Planning through latency* (ncomms/main.tex, v2) new?

**Answer.** No, not as method claims. Each one has a close precedent, and three have direct matches. What remains defensible is empirical: what latency costs learned pixel world models, and the finding that prediction through the delay, not replanning rate, dominates the loss. That is a narrower paper than v2 claims.

## Claim by claim

| v2 claim | Closest prior work | Overlap |
|---|---|---|
| **1. Latency-faithful evaluation**: the environment advances while the planner computes; d = ceil(L / Δt) | Ramstedt & Pal, *Real-Time Reinforcement Learning*, NeurIPS 2019 (arXiv 1911.04448). Thodoroff, Li & Lawrence, *Benchmarking Real-Time RL*, PMLR 181, 2022. El Asri et al., *RT-HCP*, arXiv 2509.06714 (IROS 2025), with d = floor(δ / Δt). Muppidi et al., *Finding the Time to Think*, arXiv 2606.26463 (June 2026). Staggered asynchronous inference, arXiv 2412.14355 (ICLR 2025). Anokhin et al., *Handling Delay in Real-Time RL*, ICLR 2025 | **Direct.** Thodoroff et al. already evaluated PETS, PlaNet and Dreamer with the environment running during inference, and RT-HCP uses the same definition of d. Still open: a critique of current *world-model planning* benchmarks (stable-worldmodel, World-In-World, DINO-WM / LeWM evaluations), which all pause. That is a position point, not a method. |
| **2. Committed-prefix planning**: plan from the model's prediction through the actions executing during the call | Findeisen & Allgöwer, *Computational Delay in NMPC*, IFAC 2004 (predict the state at application time). ASAP-MPC, arXiv 2402.06263 (2024): asynchronous NMPC that re-solves as soon as the last solve finishes. Chen et al., *Delay-Aware MBRL* (DATS), arXiv 2005.05440. **RT-HCP** (IROS 2025): d-step MPC with a learned model rolled through the buffered actions, CEM, and RT-TDMPC / RT-PETS baselines on a real robot. **Finding the Time to Think** (2026): MCTS simulates the committed steps. Policy side: RTC 2506.07339, RAPAC-DP 2608.15924, FutureRTC 2607.24008, WAM real-time 2608.01880, DreamZero 2602.15922 | **Direct.** The same mechanism, with learned models and sampling-based MPC, is in RT-HCP. It is textbook in NMPC. |
| **3. Exactness and the feedback-rate / mismatch decomposition** | The exactness under a perfect model is the nominal-equivalence argument of predictive delay compensation (Findeisen & Allgöwer 2004; the Smith predictor). Beger, Lin, Stanojevic & Hirche, *Optimal Delay Compensation in Networked Predictive Control*, arXiv 2512.11492 (v2 May 2026), split closed-loop error into **roll-out error** (prediction through the delay) and **open-loop error** (reusing input sequences), for linear time-invariant models | **Close.** The two terms are ours, M_cpp and F. Still open: the estimator (paired paused(r = d) arm) and its use with learned models. That is an analysis device, not a headline. |
| **4. Latency-aware configuration**: choose planner effort by the return at its own delay | Thodoroff et al. 2022, §7.3, *delay-adjusted performance* per CPU budget: algorithm rankings change with hardware. *Finding the Time to Think* (2026) learns the planning budget per state, with sim-to-hardware transfer | **Direct** for the rule. Our grid over population × iterations × GPU is a new instance, not a new idea. |

Also checked, not overlapping: *The Planning Limits of Latent World Models* (2609.39235; "ahead planning" means goal distance, not delay). Speculative asynchronous inference in DexWorldModel (2604.16484; denoising pre-computation for a world-action model). The UIUC course project 2512.17250 (learned corrector over an action queue).

## What is still defensible

1. **Empirical results on learned pixel world models across model classes**: RSSM, JEPA, TD-MPC2 and a 0.5B video diffusion model, under one planner and protocol. RT-HCP uses one hybrid physics model on one robot. Thodoroff et al. used action repetition, not compensation.
2. **The measured decomposition** (Study 1, `results/latency/summary.json`). With compensation, most of the delay loss is mismatch, not replanning rate. Committed-prefix planning removes only about 30 % of it (ratio of mismatch losses 0.64-0.73). So prediction through the delay is the bottleneck for learned world models. The pre-registered H2 (at least 50 %) failed.
3. **The diffusion latency point**: one planning call outlasts a whole episode, so paused-simulator results for diffusion world-model planners describe controllers that cannot exist.

Caution: point 2 implies that a model's accuracy through committed actions governs its real-time usefulness. That moves toward judging world models by a prediction criterion, which is the main paper's territory (metric validity). Ranking models by it would overlap the main paper.

## Options

- **A. Recast as an empirical paper** ("What inference latency costs world-model planners"), citing all of the above as the method's origin. Target a workshop, TMLR or a robotics venue. Not Nature Communications: the method contribution is gone.
- **B. Design a method aimed at the residual mismatch,** which none of the works above addresses for learned models. One direction: since MPPI iterations are sequential, re-anchor the plan on observations that arrive during the call (shrinking the predicted prefix as time passes). Check this first against Diehl's real-time iteration (its preparation and feedback phases), which is the classical form. A second direction: plan over several sampled post-delay states, so the plan is robust to prediction error through the prefix. Either needs its own novelty check and new experiments.
- **C. Choose a different side-paper topic.**

## Sources

- Ramstedt & Pal 2019: https://arxiv.org/abs/1911.04448
- Thodoroff, Li & Lawrence 2022: https://proceedings.mlr.press/v181/thodoroff22a/thodoroff22a.pdf
- RT-HCP: https://arxiv.org/abs/2509.06714
- Finding the Time to Think: https://arxiv.org/abs/2606.26463
- Staggered asynchronous inference: https://arxiv.org/abs/2412.14355
- Handling Delay in Real-Time RL: https://iclr.cc/virtual/2025/poster/29264
- Findeisen & Allgöwer 2004: https://www.sciencedirect.com/science/article/pii/S1474667017387694
- ASAP-MPC: https://arxiv.org/abs/2402.06263
- Optimal Delay Compensation in Networked Predictive Control: https://arxiv.org/abs/2512.11492
- DATS: https://arxiv.org/abs/2005.05440
- FutureRTC: https://arxiv.org/abs/2607.24008
- DreamZero: https://arxiv.org/abs/2602.15922
- RTC: https://arxiv.org/abs/2506.07339; RAPAC-DP: https://arxiv.org/abs/2608.15924; WAM real time: https://arxiv.org/abs/2608.01880
- Planning Limits of Latent World Models: https://arxiv.org/abs/2609.39235
- DexWorldModel: https://arxiv.org/abs/2604.16484

---

## Addendum (5 October 2026): candidate methods for the residual mismatch (option B)

Prior-art search for four ways to reduce the loss left after committed-prefix planning. Each verdict says how close the nearest published work is.

| Candidate | Closest prior work | Verdict |
|---|---|---|
| **C1** Finite-difference sensitivity of the sampling-based plan to the post-delay anchor, applied when the true observation arrives | Feedback-MPPI (Belvedere et al., arXiv 2506.14855, RA-L 2026): sensitivity gains for MPPI from rollout differentiation, applied as u = u* + F(x - x_ref). asNMPC (Zavala & Biegler 2009) and amsNMPC (Yang & Biegler 2013): solve at the predicted state, then a sensitivity update. AS-RTI (Nurkanović 2019). Learned action corrector over a TD-MPC2 queue (arXiv 2512.17250). Thijssen & Kappen (1406.4026): sample-based path-integral feedback | Close. **It also fails in our diagnostic**: the errors are not low-rank. |
| **C2** Multi-anchor contingent planning: plan for K hypothesised post-delay states, then select or interpolate on arrival | Multi-stage and scenario-tree MPC (Lucia et al.). Contingency and branch MPC. Particle MPC (Dyro et al., 2104.02213). Tube-MPPI and Robust MPPI (2102.09027). Steins;Gate Drive (2605.22456). DATS (2005.05440). **CLIS-MPPI** (Liu & Chen, CDC 2026; title only, no abstract or PDF public): "Closed-Loop MPPI with Initial-State Sampling under Execution Delay" | Close, and **CLIS-MPPI may be a direct match; read it before any claim**. Selection also fails in our diagnostic. |
| **Hedging** One plan scored by the average cost over K sampled post-delay anchors | **DATS** (Chen et al. 2020, 2005.05440): PETS ensemble particles pushed through the committed actions, cost averaged over them. Likely also CLIS-MPPI | Close. Our variant draws the anchors from the model's own calibration residuals, so it works for a single deterministic pretrained model with no ensemble. **It is the one that works in the diagnostic.** |
| **C3** Online calibration or bias correction of the post-delay prediction | Smith predictor. Offset-free MPC and disturbance observers (e.g. 2203.16290). Melon et al., ICRA 2021 (2104.09078). Test-time adaptation of latent world models in MPC: AdaJEPA (2606.32026), ReDRAW (2504.02252), Feedback World Model (2605.15705) | Classical. Bias correction does nothing in our diagnostic. |
| **C4** Re-anchoring on newer observations between MPPI iterations | Diehl's real-time iteration (RTI). Melon et al., ICRA 2021: "the constraint on the initial state is updated with a new prediction between each iteration of the optimizer" | Direct match. |

**Remaining novelty margin for a hedged method:**

1. Anchors come from the model's own residual bank, so no ensemble or retraining is needed. This covers deterministic latent and video world models, where DATS needs a probabilistic ensemble.
2. The mechanism: learned-model prediction errors through the committed prefix cost 1.3-3x more than random errors of the same size.
3. Evidence across model classes, including a video diffusion model.
4. Hedging may cost no latency at batch size 1, where the GPU has slack. This is pending the timing jobs.

The method idea itself is a variant of DATS-style hedging. Before any submission:

- Obtain CLIS-MPPI from the authors (Zidong Liu and Xu Chen, University of Washington) or from the CDC 2026 proceedings (December 2026).
- Compare against DATS-style ensemble hedging on at least one model.

---

## Addendum (6 October 2026): CLIS-MPPI resolved

**Source.** The CDC 2026 technical programme on PaperPlaza (last updated 5 October 2026) lists the full abstract: paper WeAT4.5, Wednesday 16 December 2026, Liu and Chen, University of Washington. There is no PDF, arXiv version or code. The authors' sibling paper, CE-MPPI (arXiv 2607.06499, IROS 2026), does not mention delay.

**What CLIS-MPPI does**, from the abstract:

- It models planning delay as a distribution over plausible execution-time initial states.
- It performs MPPI rollout optimisation from sampled initial states.
- It applies a rollout-based local feedback correction during execution.
- Validation: a real UR5e manipulator, narrow-passage task, 5 runs. Standard MPPI collided in 2 runs and CLIS-MPPI in none. Completion time fell from 25.4 s to 24.4 s and path length from 0.456 m to 0.425 m.

**Verdict.** Hedging MPPI over sampled execution-time states under delay is published, so it is CLIS-MPPI's idea, not ours. What remains distinct in our paper:

1. The hedge states come from the learned world model's own d-step prediction residuals (model error), rather than from an assumed delay distribution with a known robot model.
2. The diagnosis: learned-model errors through the delay are value-aligned and leave about 70 % of the mismatch after committed-prefix planning.
3. Batched anchor scoring costs no latency at batch size 1.
4. The pre-registered evaluation over 24 pixel world models in three families, plus a diffusion world model.
5. The wall-clock finding: the delay must cover the whole loop.

The method claim is now "residual-calibrated hedging for learned world models, building on CLIS-MPPI". The manuscript credits CLIS-MPPI in the Introduction, the contributions list and the Discussion (6 October 2026).

---

## Addendum (11 October 2026): DATS-style ensemble comparison done

**Why.** The outstanding check above, also raised in reviewer feedback on the manuscript.

**Design** (`EXPERIMENT_PLAN_SIDE.md` Section 14, pre-registered). DATS-style ensemble hedging uses the three independently trained models of each cell. Each member keeps its own particle (TS-inf), because the members' latent spaces differ. Candidate costs are averaged over members. The ensemble ran at matched population and at matched batch-1 latency. Also run: particle anchors from the RSSM's own stochastic transition, and ensemble + RCH.

**Result.**

1. **Pre-registered:** RCH is below the latency-matched ensemble at d = 4 (-0.042, Holm p = 0.0008). The other three differences favour the ensemble, with intervals including zero.
2. **Post hoc:** the ensemble's lead is present without delay (+0.066 paused), and its mismatch loss equals single-model committed-prefix planning's. RCH reduces mismatch loss, and adds to the ensemble at d = 8 (+0.033).
3. **Particle anchors** on the stochastic RSSM match or beat RCH. Sampling the model's own transition is an alternative hedge source for stochastic models.

**Consequence for the claims.**

- RCH is not claimed to beat ensemble hedging on return.
- The distinct claims that remain:
  1. Hedging for a single, possibly deterministic, pretrained world model, with no extra training.
  2. It is the component that reduces the delay-specific mismatch loss.
  3. It combines with an ensemble at the longer delay.
- The ensemble's costs: 3x training, weights and filtering; 2.9x call latency at matched population.
- The manuscript states the ensemble result in the abstract, Results, Discussion and Methods.

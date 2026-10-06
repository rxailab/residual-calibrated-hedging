"""Delay-in-decisions schedules and committed-prefix MPPI (side paper, Methods).

Family-agnostic: a run needs a batched cost function and an environment batch
with `latent()` and `step(actions)`.  The same code runs the G0 exactness test
on an exact linear model (tests/test_latency.py) and every closed-loop study.

Arms (E episodes in lockstep, one delay d for all):
  paused<r>   the simulator waits for the planner; r actions executed per plan
              (paused1 is the standard report).
  naive<d>    asynchronous; plans from the latent at the start of a call as if
              no time passes, executes plan positions 0:d one call later.
  aligned<d>  asynchronous; same plan, executes positions d:2d (time-aligned).
  cpp<d>      asynchronous; every candidate's first d actions are the actions
              committed for execution during the call, so the model predicts
              through the delay; executes the first d free actions.
Asynchronous arms start with one plan from the initial observation with the
simulator paused (the robot is at rest before the episode), whose first d
actions are committed for the first interval.  Their candidates are d + H
long, so naive, aligned and cpp make identical model calls; paused uses H.
"""
from __future__ import annotations

import dataclasses
import re
import time

import torch


@dataclasses.dataclass
class MPPIConfig:
    population: int = 512
    iterations: int = 6
    temperature: float = 0.5
    init_std: float = 1.0
    min_std: float = 0.05
    momentum: float = 0.1
    low: float = -1.0
    high: float = 1.0


class PrefixMPPI:
    """Batched MPPI over E episodes (the update of planners/mppi.py and
    experiments/dmc/evaluate.mppi_step), with an optional committed prefix.

    cost_fn(seq) -> (E * P,) costs for candidate sequences seq of shape
    (E * P, L, A), episode-major, from each episode's current latent (the
    caller binds the latent).  Only the free part is sampled and updated; the
    prefix enters the cost only.  A non-finite cost gets weight zero, and an
    episode with no finite candidate gets equal weights (evaluate_diffusion D11).
    """

    def __init__(self, cfg: MPPIConfig, gen: torch.Generator, dev, dtype=torch.float64):
        self.cfg, self.gen, self.dev, self.dtype = cfg, gen, dev, dtype
        self.n_bad = 0
        self.n_all = 0

    def __call__(self, cost_fn, warm, prefix=None):
        c = self.cfg
        E, L, A = warm.shape
        P = c.population
        mu = warm.clone()
        std = torch.full_like(mu, c.init_std)
        for _ in range(c.iterations):
            eps = torch.randn((E, P, L, A), generator=self.gen, device=self.dev, dtype=self.dtype)
            free = (mu[:, None] + std[:, None] * eps).clamp(c.low, c.high)
            seq = free if prefix is None else torch.cat(
                [prefix[:, None].expand(E, P, *prefix.shape[1:]), free], 2)
            J = torch.as_tensor(cost_fn(seq.reshape(E * P, -1, A)), dtype=self.dtype,
                                device=self.dev).reshape(E, P)
            fin = torch.isfinite(J)
            self.n_bad += int((~fin).sum()); self.n_all += fin.numel()
            mn = torch.where(fin, J, torch.full_like(J, float("inf"))).min(1, keepdim=True).values
            mn = torch.where(torch.isfinite(mn), mn, torch.zeros_like(mn))
            w = torch.where(fin, torch.exp(-(torch.where(fin, J, mn) - mn) / max(c.temperature, 1e-8)),
                            torch.zeros_like(J))
            sw = w.sum(1, keepdim=True)
            w = torch.where(sw > 0, w / sw.clamp_min(1e-300), torch.full_like(w, 1.0 / P))
            new_mu = torch.einsum("ep,epla->ela", w, free)
            new_std = torch.sqrt(torch.einsum("ep,epla->ela", w, (free - new_mu[:, None]) ** 2))
            mu = (1 - c.momentum) * new_mu + c.momentum * mu
            std = new_std.clamp_min(c.min_std)
        return mu


def shift(mu, k):
    """Warm start for a plan that starts k decisions later: drop the first k
    rows, pad k zero rows at the end."""
    out = torch.roll(mu, -k, 1)
    out[:, -k:] = 0.0
    return out


def pad(mu, k):
    return torch.cat([mu, torch.zeros_like(mu[:, :k])], 1)


def parse_arm(arm):
    m = re.fullmatch(r"(paused|naive|aligned|cpp)(\d+)", arm)
    if not m:
        raise ValueError(f"bad arm {arm!r}")
    return m.group(1), int(m.group(2))


class Timer:
    def __init__(self, dev):
        self.cuda = str(dev).startswith("cuda")
        self.calls, self.total = 0, 0.0
        self.per_call = []

    def __enter__(self):
        if self.cuda:
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()

    def __exit__(self, *exc):
        if self.cuda:
            torch.cuda.synchronize()
        dt = time.perf_counter() - self.t0
        self.calls += 1; self.total += dt; self.per_call.append(dt)


def run_schedule(arm, H, T, planner, make_cost, env, A, timer=None):
    """Run one arm for T decisions on env (E episodes in lockstep).

    make_cost(Z) -> cost_fn bound to latents Z; planner = PrefixMPPI;
    env.latent() -> current latents, env.step(actions (E, A) tensor).
    Plan computation is timed with `timer` (CUDA-synchronised); actions
    returned by a call are copied to the host inside the timed region by
    env.step's caller, so the timer covers the plan only.
    Returns the number of planning calls.
    """
    kind, k = parse_arm(arm)
    E = env.E
    dev, dt = planner.dev, planner.dtype
    timer = timer or Timer(dev)

    def plan(warm, prefix=None):
        Z = env.latent()
        with timer:
            mu = planner(make_cost(Z), warm, prefix)
        return mu

    zeros = torch.zeros((E, H, A), dtype=dt, device=dev)
    t = 0
    if kind == "paused":
        r = k
        ws = zeros
        while t < T:
            mu = plan(ws)
            for j in range(r):
                if t >= T:
                    break
                env.step(mu[:, j]); t += 1
            ws = shift(mu, r)
        return timer.calls

    d = k
    if kind != "cpp" and d > H:
        raise ValueError("time-aligned and naive arms need d <= H")
    mu0 = plan(zeros)                                    # pre-episode plan, simulator paused
    committed = mu0[:, :d].clone()
    w = shift(mu0, d) if kind == "cpp" else pad(mu0, d)
    while t < T:
        nxt = None
        if t + d < T:                                   # a later interval exists: plan it
            if kind == "cpp":
                mu = plan(w, committed)
                nxt = mu[:, :d].clone()
            else:
                mu = plan(w)
                nxt = (mu[:, d:2 * d] if kind == "aligned" else mu[:, :d]).clone()
            w = shift(mu, d)
        for j in range(d):                              # the committed actions run during the call
            if t >= T:
                break
            env.step(committed[:, j]); t += 1
        committed = nxt
    return timer.calls

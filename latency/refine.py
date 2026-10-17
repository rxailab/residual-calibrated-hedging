"""Hedge-then-refine (plan Section 17A): RCH whose last MPPI iteration is re-anchored on a fresher
observation, at the same compute and latency as RCH.

A call of I = 6 iterations that starts at decision t and whose plan takes effect at t + d is split:
  1. iterations 1 .. I - 1 from the latent at t, with the d committed actions as the prefix and
     anchors from the d-step residual bank (RCH, latency/hedge.py);
  2. the last iteration from the latent observed at t + d - q, q = ceil(d / I) (that iteration's
     share of the call's latency, rounded up), with the last q committed actions as the prefix and
     anchors from a q-step bank, starting from the mean and standard deviation of step 1.
With K = 1 it is re-anchored committed-prefix planning.  The plan still takes effect at t + d.

`run_reanchor` mirrors latency/schedule.run_schedule for the cpp arm; `mppi_run` is
PrefixMPPI.__call__ with a given initial mean, standard deviation and iteration count (the update
equations are copied unchanged; PrefixMPPI itself is not modified).
"""
from __future__ import annotations

import math

import torch

from latency.schedule import PrefixMPPI, Timer, shift


def mppi_run(planner: PrefixMPPI, cost_fn, mean, std, iters, prefix=None):
    """PrefixMPPI.__call__ from (mean, std) for `iters` iterations; returns (mu, std)."""
    c = planner.cfg
    E, L, A = mean.shape
    P = c.population
    mu, std = mean.clone(), std.clone()
    for _ in range(iters):
        eps = torch.randn((E, P, L, A), generator=planner.gen, device=planner.dev, dtype=planner.dtype)
        free = (mu[:, None] + std[:, None] * eps).clamp(c.low, c.high)
        seq = free if prefix is None else torch.cat([prefix[:, None].expand(E, P, *prefix.shape[1:]), free], 2)
        J = torch.as_tensor(cost_fn(seq.reshape(E * P, -1, A)), dtype=planner.dtype, device=planner.dev).reshape(E, P)
        fin = torch.isfinite(J)
        planner.n_bad += int((~fin).sum()); planner.n_all += fin.numel()
        mn = torch.where(fin, J, torch.full_like(J, float("inf"))).min(1, keepdim=True).values
        mn = torch.where(torch.isfinite(mn), mn, torch.zeros_like(mn))
        w = torch.where(fin, torch.exp(-(torch.where(fin, J, mn) - mn) / max(c.temperature, 1e-8)), torch.zeros_like(J))
        sw = w.sum(1, keepdim=True)
        w = torch.where(sw > 0, w / sw.clamp_min(1e-300), torch.full_like(w, 1.0 / P))
        new_mu = torch.einsum("ep,epla->ela", w, free)
        new_std = torch.sqrt(torch.einsum("ep,epla->ela", w, (free - new_mu[:, None]) ** 2))
        mu = (1 - c.momentum) * new_mu + c.momentum * mu
        std = new_std.clamp_min(c.min_std)
    return mu, std


def run_reanchor(d, H, T, planner, make_cost_d, make_cost_q, make_cost_paused, env, A, timer=None, iters_ref=1):
    """Hedge-then-refine for T decisions on env (E episodes in lockstep); returns the number of calls.
    make_cost_d(Z) scores [d committed; H free] candidates from Z, make_cost_q(Z) scores
    [q committed; H free] candidates, make_cost_paused(Z) scores H free candidates (pre-episode plan)."""
    c = planner.cfg
    I = c.iterations
    q = max(1, math.ceil(d * iters_ref / I))
    E = env.E
    dev, dt = planner.dev, planner.dtype
    timer = timer or Timer(dev)
    zeros = torch.zeros((E, H, A), dtype=dt, device=dev)
    with timer:                                              # pre-episode plan, simulator paused
        mu0 = planner(make_cost_paused(env.latent()), zeros)
    committed = mu0[:, :d].clone()
    w = shift(mu0, d)
    t = 0
    while t < T:
        plan_next = t + d < T                                # a later interval exists: plan it
        if plan_next:
            with timer:                                      # step 1: from the latent at t
                mu, std = mppi_run(planner, make_cost_d(env.latent()), w, torch.full_like(w, c.init_std),
                                   I - iters_ref, committed)
        for j in range(d - q):                               # the committed actions run meanwhile
            if t >= T:
                break
            env.step(committed[:, j]); t += 1
        nxt = None
        if plan_next:                                        # step 2: re-anchored on the latent at t + d - q
            with timer:
                mu, std = mppi_run(planner, make_cost_q(env.latent()), mu, std, iters_ref, committed[:, d - q:])
            nxt = mu[:, :d].clone()
            w = shift(mu, d)
        for j in range(d - q, d):
            if t >= T:
                break
            env.step(committed[:, j]); t += 1
        committed = nxt
    return timer.calls

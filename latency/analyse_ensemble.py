"""Analysis of the ensemble and particle hedging baselines (paper/docs/EXPERIMENT_PLAN_SIDE.md,
Section 14), with the rules written before the runs.

Arms (all at d = 4 and 8; 50 episodes from seeds 950,000-950,049):
  CPP       hcpp<d>k1, single model, population 512, 6 iterations (24 models)
  RCH       hcpp<d>k4, single model, population 512, 6 iterations (24 models)
  E-pop     ens<d>, three-model ensemble, population 512, 6 iterations (8 cells)
  E-lat     ens<d>, three-model ensemble, population 512, 2 iterations (8 cells)
  E+RCH     ens<d>k4, ensemble with four anchors per member, 6 iterations (8 cells)
  particles pcpp<d>k4, RSSM only, population 512, 6 iterations (9 models, 3 cells)

Ensemble vs a single-model arm, cell by cell: the ensemble's normalised return on
each episode minus each single model's return on the same initial state, so a
cell gives a (3 models x 50 episodes) array of paired differences.  Cells are
weighted equally.  Hierarchical bootstrap as in latency/analyse_hedge_full.py
(models within cell, then episodes within model), 10,000 resamples.  The
ensemble is one model set per cell, so the intervals do not include variation
between ensembles.

Primary, two-sided, Holm over the four contrasts:
  RCH vs E-pop and RCH vs E-lat, each at d = 4 and 8.
Secondary (two-sided): E-pop vs CPP, E-lat vs CPP; E+RCH vs E-pop and vs RCH;
particles vs RCH and vs CPP (3 RSSM cells).
Writes results/latency/ensemble.json.
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from latency.analyse import norm  # noqa: E402
from latency.analyse_hedge import CELLS  # noqa: E402
from latency.analyse_hedge_full import holm  # noqa: E402
from paths import RESULTS  # noqa: E402

SEEDS = ("seed0", "seed1", "seed2")
B = 10_000
P6, P2 = "pop512_it6", "pop512_it2"


def single(cell, cfg, arm):
    xs = []
    for s in SEEDS:
        f = RESULTS / "latency" / "dmc" / cell / s / cfg / f"{arm}.json"
        if not f.exists():
            return None
        xs.append(norm(cell.split("/")[1], json.load(open(f))["returns"]))
    return np.stack(xs)                                             # (3, 50)


def ens(cell, cfg, arm):
    f = RESULTS / "latency" / "ensemble" / cell / cfg / f"{arm}.json"
    return norm(cell.split("/")[1], json.load(open(f))["returns"])[None] if f.exists() else None   # (1, 50)


def contrast(a, b, rng):
    """a, b: (loader, cfg, arm).  Paired differences per cell, broadcast to the
    single-model axis when one side is the ensemble."""
    diffs = []
    for c in CELLS:
        x, y = a[0](c, *a[1:]), b[0](c, *b[1:])
        if x is None or y is None:
            continue
        diffs.append(x - y)                                         # (3, 50) or (1, 50)
    est = float(np.mean([x.mean() for x in diffs]))
    bs = np.zeros(B)
    for x in diffs:
        m, n = x.shape
        mi = rng.integers(0, m, (B, m)); ei = rng.integers(0, n, (B, m, n))
        bs += x[mi[..., None], ei].mean((1, 2))
    bs /= len(diffs)
    p = float(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())))
    return {"est": est, "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "p_two_sided": p, "cells": len(diffs)}


def combo(*terms):
    """A loader for a signed sum of arms, terms = (sign, loader, cfg, arm); the
    ensemble's (1, 50) arrays broadcast against the single models' (3, 50)."""
    def load(c, *_):
        xs = [t[1](c, *t[2:]) for t in terms]
        return None if any(x is None for x in xs) else sum(t[0] * x for t, x in zip(terms, xs))
    return load


def level(loader, cfg, arm):
    """Equal-cell-weight mean normalised return of one arm."""
    xs = [loader(c, cfg, arm) for c in CELLS]
    xs = [x for x in xs if x is not None]
    return {"mean": float(np.mean([x.mean() for x in xs])), "cells": len(xs)} if xs else None


def main():
    rng = np.random.default_rng(20261011)
    S, E = single, ens
    out = {"levels": {}, "primary": {}, "secondary": {}}
    for d in (4, 8):
        out["levels"][f"d{d}"] = {k: level(*v) for k, v in {
            "CPP": (S, P6, f"hcpp{d}k1"), "RCH": (S, P6, f"hcpp{d}k4"), "E-pop": (E, P6, f"ens{d}"),
            "E-lat": (E, P2, f"ens{d}"), "E+RCH": (E, P6, f"ens{d}k4"), "particles": (S, P6, f"pcpp{d}k4")}.items()}
        out["primary"][f"RCH_vs_Epop_d{d}"] = contrast((S, P6, f"hcpp{d}k4"), (E, P6, f"ens{d}"), rng)
        out["primary"][f"RCH_vs_Elat_d{d}"] = contrast((S, P6, f"hcpp{d}k4"), (E, P2, f"ens{d}"), rng)
        sec = out["secondary"]
        sec[f"Epop_vs_CPP_d{d}"] = contrast((E, P6, f"ens{d}"), (S, P6, f"hcpp{d}k1"), rng)
        sec[f"Elat_vs_CPP_d{d}"] = contrast((E, P2, f"ens{d}"), (S, P6, f"hcpp{d}k1"), rng)
        if level(E, P6, f"ens{d}k4"):
            sec[f"EplusRCH_vs_Epop_d{d}"] = contrast((E, P6, f"ens{d}k4"), (E, P6, f"ens{d}"), rng)
            sec[f"EplusRCH_vs_RCH_d{d}"] = contrast((E, P6, f"ens{d}k4"), (S, P6, f"hcpp{d}k4"), rng)
        if level(S, P6, f"pcpp{d}k4"):
            sec[f"particles_vs_RCH_d{d}"] = contrast((S, P6, f"pcpp{d}k4"), (S, P6, f"hcpp{d}k4"), rng)
            sec[f"particles_vs_CPP_d{d}"] = contrast((S, P6, f"pcpp{d}k4"), (S, P6, f"hcpp{d}k1"), rng)
            sec[f"RCH_vs_CPP_RSSM_d{d}"] = contrast(
                (lambda c, *k: S(c, *k) if c.startswith("rssm") else None, P6, f"hcpp{d}k4"), (S, P6, f"hcpp{d}k1"), rng)
    pr = out["primary"]
    for k, v in zip(pr, holm([pr[k]["p_two_sided"] for k in pr])):
        pr[k]["p_holm"] = v
    # post hoc (plan Section 14, written after the primary result): iteration count, and RSSM-only levels;
    # a separate stream, so the numbers above do not change
    rng2 = np.random.default_rng(20261012)
    rssm = lambda c, *k: S(c, *k) if c.startswith("rssm") else None
    out["posthoc"], out["levels_rssm"] = {}, {}
    for d in (4, 8):
        out["levels_rssm"][f"d{d}"] = {k: level(rssm, P6, a) for k, a in (("CPP", f"hcpp{d}k1"), ("RCH", f"hcpp{d}k4"),
                                                                           ("particles", f"pcpp{d}k4"))}
        ph = out["posthoc"]
        ph[f"Elat_vs_Epop_d{d}"] = contrast((E, P2, f"ens{d}"), (E, P6, f"ens{d}"), rng2)
        if level(S, P2, f"hcpp{d}k4") and level(S, P2, f"hcpp{d}k4")["cells"] == len(CELLS):
            out["levels"][f"d{d}"]["RCH it2"] = level(S, P2, f"hcpp{d}k4")
            out["levels"][f"d{d}"]["CPP it2"] = level(S, P2, f"hcpp{d}k1")
            ph[f"RCHit2_vs_Elat_d{d}"] = contrast((S, P2, f"hcpp{d}k4"), (E, P2, f"ens{d}"), rng2)
            ph[f"RCHit2_vs_RCHit6_d{d}"] = contrast((S, P2, f"hcpp{d}k4"), (S, P6, f"hcpp{d}k4"), rng2)
            ph[f"CPPit2_vs_CPPit6_d{d}"] = contrast((S, P2, f"hcpp{d}k1"), (S, P6, f"hcpp{d}k1"), rng2)
            ph[f"RCHit2_vs_CPPit2_d{d}"] = contrast((S, P2, f"hcpp{d}k4"), (S, P2, f"hcpp{d}k1"), rng2)
    # second post-hoc check: the ensemble without delay, and each planner's mismatch loss against
    # its own paused planner, M(d) = R(paused_d) - R(planner_d)
    if level(E, P6, "enspaused1") and level(E, P6, "enspaused1")["cells"] == len(CELLS):
        dec = out["decomposition"] = {}
        ZERO = (lambda c, *_: np.zeros((1, 50)), None, None)          # contrast against 0: a level with its interval
        dec["E_vs_single_paused1"] = contrast((E, P6, "enspaused1"), (S, P6, "paused1"), rng2)
        dec["Elat_vs_single_paused1"] = contrast((E, P2, "enspaused1"), (S, P6, "paused1"), rng2)
        for d in (4, 8):
            dec[f"levels_d{d}"] = {k: level(*v) for k, v in {
                "single paused1": (S, P6, "paused1"), f"single paused{d}": (S, P6, f"paused{d}"),
                "E paused1": (E, P6, "enspaused1"), f"E paused{d}": (E, P6, f"enspaused{d}"),
                "E-lat paused1": (E, P2, "enspaused1"), f"E-lat paused{d}": (E, P2, f"enspaused{d}")}.items()}
            dec[f"E_vs_single_paused{d}"] = contrast((E, P6, f"enspaused{d}"), (S, P6, f"paused{d}"), rng2)
            M = {"CPP": combo((1, S, P6, f"paused{d}"), (-1, S, P6, f"hcpp{d}k1")),
                 "RCH": combo((1, S, P6, f"paused{d}"), (-1, S, P6, f"hcpp{d}k4")),
                 "E-pop": combo((1, E, P6, f"enspaused{d}"), (-1, E, P6, f"ens{d}")),
                 "E-lat": combo((1, E, P2, f"enspaused{d}"), (-1, E, P2, f"ens{d}")),
                 "E+RCH": combo((1, E, P6, f"enspaused{d}"), (-1, E, P6, f"ens{d}k4"))}
            for k, f in M.items():
                dec[f"M_{k}_d{d}"] = contrast((f, None, None), ZERO, rng2)
            Fs = {"single": combo((1, S, P6, "paused1"), (-1, S, P6, f"paused{d}")),
                  "E": combo((1, E, P6, "enspaused1"), (-1, E, P6, f"enspaused{d}"))}
            for k, f in Fs.items():
                dec[f"F_{k}_d{d}"] = contrast((f, None, None), ZERO, rng2)
            dec[f"M_RCH_minus_M_Epop_d{d}"] = contrast((M["RCH"], None, None), (M["E-pop"], None, None), rng2)
            dec[f"M_RCH_minus_M_Elat_d{d}"] = contrast((M["RCH"], None, None), (M["E-lat"], None, None), rng2)
            dec[f"M_CPP_minus_M_Epop_d{d}"] = contrast((M["CPP"], None, None), (M["E-pop"], None, None), rng2)
    tim = {}
    for f in sorted((RESULTS / "latency" / "timing_ensemble").glob("*/*.json")):
        tim[f"{f.parent.name}/{f.stem}"] = json.load(open(f))
    out["timing_files"] = sorted(tim)
    (RESULTS / "latency" / "ensemble.json").write_text(json.dumps(out, indent=1))
    for d, lv in out["levels"].items():
        print(d, "  ".join(f"{k} {v['mean']:.3f} ({v['cells']})" for k, v in lv.items() if v))
    print("RSSM only", out["levels_rssm"])
    for sect in ("primary", "secondary", "posthoc", "decomposition"):
        print(sect)
        for k, v in out.get(sect, {}).items():
            if "est" not in v:
                print(f"  {k}: " + "  ".join(f"{a} {b['mean']:.3f}" for a, b in v.items() if b)); continue
            ph = f", Holm p = {v['p_holm']:.4f}" if "p_holm" in v else ""
            print(f"  {k:24s} {v['est']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]  two-sided p = "
                  f"{v['p_two_sided']:.4f}{ph}  ({v['cells']} cells)")
    for k, t in tim.items():
        print(k, t["gpu"], [(m["run"], m["params"]) for m in t["members"]])
        for r in t["results"]:
            print(f"   d{r['d']} {r['config']:16s} it{r['iters']}  L50 {r['L50_s'] * 1e3:6.1f} ms  "
                  f"L95 {r['L95_s'] * 1e3:6.1f} ms  peak +{r['peak_extra_bytes'] / 2 ** 20:6.0f} MiB")
        print("   filter", {n: round(v["L50_s"] * 1e3, 2) for n, v in t["filter"].items()})


if __name__ == "__main__":
    main()

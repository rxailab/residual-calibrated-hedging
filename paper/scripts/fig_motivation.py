"""Figure 1: amortised planning cost versus the duration of one planning call
(reanalysis of the diffusion configuration records, evidence/configuration_summary.csv).

Redesign of 6 October 2026 (Claude Design handoff, figures/cost_vs_latency.svg):
1440 x 600 px, two panels, bars grouped by replanning interval, the simulated duration
of a whole episode as a shaded band in panel b.
Run: /mmfs1/storage/users/xiar3/envs/figenv/bin/python scripts/fig_motivation.py
"""
import csv
import math

import figstyle as F

rows = list(csv.DictReader(open(F.ROOT / "evidence" / "configuration_summary.csv")))
C = [float(r["planning_seconds_per_agent_decision"]) for r in rows]
B = [float(r["mean_batch_planning_call_seconds"]) for r in rows]
S = [int(r["sampler_steps"]) for r in rows]
RP = [int(r["replan_decisions"]) for r in rows]
# simulated duration of one 500-decision episode: action repeat 2 x control timestep
EPISODE_S = {"cartpole-swingup": 500 * 2 * 0.01, "walker-walk": 500 * 2 * 0.025, "cup-catch": 500 * 2 * 0.02}
ep_lo, ep_hi = min(EPISODE_S.values()), max(EPISODE_S.values())

Y0, YTOP = 440, 100                      # plot baseline and top, px


def panel(x0, x1, values, ymax, step, fmt, title_x, letter, title, ylab, ylab_x, note, band=None, headroom=0):
    """ymax: last tick; headroom: extra data units above it (room for notes and the legend)."""
    out = [F.panel_title(title_x, 52, letter, title)]
    sy = (Y0 - YTOP) / (ymax + headroom)
    y = lambda v: Y0 - v * sy
    for v in range(0, int(ymax) + 1, step):
        out.append(F.line(x0, y(v), x1, y(v), F.GRID))
        out.append(F.text(x0 - 14, y(v), str(v), size=19, fill=F.MUTED, anchor="end", baseline="middle"))
    if band:
        lo, hi = band
        out.append(F.rect(x0, y(hi), x1 - x0, y(lo) - y(hi), F.ORANGE, fill_opacity="0.22"))
        out.append(F.line(x0, y(hi), x1, y(hi), F.ORANGE, 1.2, "5 4"))
        out.append(F.line(x0, y(lo), x1, y(lo), F.ORANGE, 1.2, "5 4"))
    slot = (x1 - x0) / len(values)
    bw = 0.58 * slot
    centers = [x0 + slot * (i + 0.5) for i in range(len(values))]
    for c, v, s in zip(centers, values, S):
        out.append(F.rect(c - bw / 2, y(v), bw, Y0 - y(v), F.NAVY, rx=2))
        out.append(F.text(c, y(v) - 10, fmt.format(v), size=18, anchor="middle", weight=500))
        out.append(F.text(c, 468, f"s = {s}", size=19, fill=F.MUTED, anchor="middle"))
    out.append(F.line(x0, Y0, x1, Y0, F.AXIS, 1.5))
    for r in sorted(set(RP)):
        idx = [i for i, q in enumerate(RP) if q == r]
        a, b = centers[idx[0]] - bw / 2, centers[idx[-1]] + bw / 2
        out.append(F.line(a, 492, b, 492, F.AXIS, 1.2))
        out.append(F.text((a + b) / 2, 518, f"replan every {r} decisions", size=19, anchor="middle"))
    out.append(F.text(ylab_x, 270, ylab, size=21, anchor="middle", rotate=-90))
    out.append(F.text((x0 + x1) / 2, 118, note, size=18, fill=F.MUTED, anchor="middle"))
    return out


ratio_c = max(C) / min(C)
ratio_b = max(B) / min(B)
body = []
body += panel(130, 660, C, 3 * math.ceil(max(C) / 3), 3, "{:.1f}", 60, "a", "Amortised planning cost",
              "planning seconds per agent decision", 58, f"{ratio_c:.1f}× lower across the six configurations")
tick_b = 50 * math.ceil(max(B) / 50)
body += panel(900, 1410, B, tick_b, 50, "{:.0f}", 830, "b", "Duration of one planning call",
              "seconds per batched call (5 episodes)", 828,
              f"{ratio_b:.1f}× lower at most; planning less often changes nothing", band=(ep_lo, ep_hi),
              headroom=0.16 * tick_b)
body.append(F.rect(1000, 138, 22, 12, F.ORANGE, stroke=F.ORANGE, stroke_width="1", fill_opacity="0.35"))
body.append(F.text(1032, 144, f"one whole episode ({ep_lo:.0f}–{ep_hi:.0f} s simulated)", size=18,
                   fill=F.ORANGE_TEXT, baseline="middle"))
body.append(F.text(770, 560, "sampler steps s; one planning call serves r decisions", size=18, fill=F.MUTED,
                   anchor="middle"))
F.render(F.svg(1440, 600, body), "cost_vs_latency")

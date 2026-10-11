"""Figure 1: amortised planning cost versus the duration of one planning call
(reanalysis of the diffusion configuration records, evidence/configuration_summary.csv).

Redesign of 6 October 2026 (Claude Design handoff, figures/cost_vs_latency.svg):
1440 x 600 px, two panels, bars grouped by replanning interval, the simulated duration
of a whole episode as a shaded band in panel b.
Revision of 8 October 2026: each bar is the mean over the three tasks, and the per-task
values (evidence/pilot_records.json) are overlaid as dots, left to right in TASKS order.
All text is 8 pt at the 180-mm print width (figstyle.TEXT), so the canvas is 1440 x 690 px,
the notes sit above the plots, and the definitions of s and r moved to the figure legend.
Run: /mmfs1/storage/users/xiar3/envs/figenv/bin/python scripts/fig_motivation.py
"""
import csv
import json
import math

import figstyle as F

rows = list(csv.DictReader(open(F.ROOT / "evidence" / "configuration_summary.csv")))
C = [float(r["planning_seconds_per_agent_decision"]) for r in rows]
B = [float(r["mean_batch_planning_call_seconds"]) for r in rows]
S = [int(r["sampler_steps"]) for r in rows]
RP = [int(r["replan_decisions"]) for r in rows]
# per-task values behind each bar, derived as in analyse_pilot.py (5 episodes x 500 decisions)
TASKS = ["cartpole-swingup", "walker-walk", "cup-catch"]
RECS = json.loads((F.ROOT / "evidence" / "pilot_records.json").read_text(encoding="utf-8"))["records"]
per_task = lambda s, r: [next(x for x in RECS if x["task"] == t and x["eval_settings"]["steps"] == s
                              and x["eval_settings"]["replan_blocks"] * 4 == r) for t in TASKS]
PC = [[x["t_u1_plan_s"] / 2500 for x in per_task(s, r)] for s, r in zip(S, RP)]
PB = [[x["t_u1_plan_s"] / math.ceil(500 / r) for x in per_task(s, r)] for s, r in zip(S, RP)]
assert all(math.isclose(sum(p) / len(p), v, rel_tol=1e-9) for p, v in zip(PC + PB, C + B))
# simulated duration of one 500-decision episode: action repeat 2 x control timestep
EPISODE_S = {"cartpole-swingup": 500 * 2 * 0.01, "walker-walk": 500 * 2 * 0.025, "cup-catch": 500 * 2 * 0.02}
ep_lo, ep_hi = min(EPISODE_S.values()), max(EPISODE_S.values())

W, H = 1440, 690
Y0, YTOP = 588, 172                      # plot baseline and top (last tick), px


def dot(x, y):
    """One overlaid data point: white fill with a dark ring, visible on the bars and on white."""
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5.5" fill="#fff" stroke="{F.INK}" stroke-width="1.6"></circle>'


def panel(x0, x1, values, points, ymax, step, fmt, title_x, letter, title, ylab, ylab_x, notes):
    """points: per-task values behind each bar; ymax: last tick; notes: lines above the plot."""
    out = [F.panel_title(title_x, 44, letter, title)]
    sy = (Y0 - YTOP) / ymax
    y = lambda v: Y0 - v * sy
    for v in range(0, int(ymax) + 1, step):
        out.append(F.line(x0, y(v), x1, y(v), F.GRID))
        out.append(F.text(x0 - 14, y(v), str(v), fill=F.MUTED, anchor="end", baseline="middle"))
    slot = (x1 - x0) / len(values)
    bw = 0.58 * slot
    centers = [x0 + slot * (i + 0.5) for i in range(len(values))]
    for c, v, s, pts in zip(centers, values, S, points):
        out.append(F.rect(c - bw / 2, y(v), bw, Y0 - y(v), F.NAVY, rx=2))
        out += [dot(c + (k - (len(pts) - 1) / 2) * bw / 4, y(p)) for k, p in enumerate(pts)]
        out.append(F.text(c, y(max(v, *pts)) - 16, fmt.format(v), anchor="middle", weight=500))
        out.append(F.text(c, Y0 + 32, f"s = {s}", fill=F.MUTED, anchor="middle"))
    out.append(F.line(x0, Y0, x1, Y0, F.AXIS, 1.5))
    for r in sorted(set(RP)):
        idx = [i for i, q in enumerate(RP) if q == r]
        a, b = centers[idx[0]] - bw / 2, centers[idx[-1]] + bw / 2
        out.append(F.line(a, Y0 + 52, b, Y0 + 52, F.AXIS, 1.2))
        out.append(F.text((a + b) / 2, Y0 + 82, f"replan every {r} decisions", anchor="middle"))
    out.append(F.text(ylab_x, (Y0 + YTOP) / 2, ylab, anchor="middle", rotate=-90))
    out += [F.text((x0 + x1) / 2, 88 + 28 * i, note, fill=F.MUTED, anchor="middle") for i, note in enumerate(notes)]
    return out


def band(x0, x1, ymax, lo, hi):
    """Shaded band for the simulated duration of a whole episode, drawn under the bars."""
    sy = (Y0 - YTOP) / ymax
    y = lambda v: Y0 - v * sy
    return [F.rect(x0, y(hi), x1 - x0, y(lo) - y(hi), F.ORANGE, fill_opacity="0.22"),
            F.line(x0, y(hi), x1, y(hi), F.ORANGE, 1.2, "5 4"), F.line(x0, y(lo), x1, y(lo), F.ORANGE, 1.2, "5 4")]


ratio_c = max(C) / min(C)
ratio_b = max(B) / min(B)
AX0, AX1, BX0, BX1 = 100, 660, 868, 1428
tick_b = 50 * math.ceil(max(B) / 50)
body = band(BX0, BX1, tick_b, ep_lo, ep_hi)
body += panel(AX0, AX1, C, PC, 3 * math.ceil(max(C) / 3), 3, "{:.1f}", 30, "a", "Amortised planning cost",
              "planning seconds per agent decision", 34, [f"{ratio_c:.1f}× lower across the six configurations"])
body += panel(BX0, BX1, B, PB, tick_b, 50, "{:.0f}", 776, "b", "Duration of one planning call",
              "seconds per batched call (5 episodes)", 790,
              [f"{ratio_b:.1f}× lower at most;", "planning less often changes nothing"])
body.append(F.rect(BX0 + 72, 137, 24, 14, F.ORANGE, stroke=F.ORANGE, stroke_width="1", fill_opacity="0.35"))
body.append(F.text(BX0 + 106, 144, f"one whole episode ({ep_lo:.0f}–{ep_hi:.0f} s simulated)",
                   fill=F.ORANGE_TEXT, baseline="middle"))
F.render(F.svg(W, H, body), "cost_vs_latency")

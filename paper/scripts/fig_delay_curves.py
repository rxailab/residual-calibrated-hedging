"""Figure 3: closed-loop return against delay (Study 1, pooled over eligible cells) and the
split of the delay loss into feedback-rate and mismatch terms (results/latency/summary.json).

Redesign of 6 October 2026 (Claude Design handoff, figures/delay_curves.svg):
1440 x 600 px; panel a with direct end-of-line labels and 95 % bands, panel b grouped bars
with 95 % intervals.
Revision of 8 October 2026: panel b overlays the eight eligible family-task cells as dots
(each bar is their equal-weight mean, as in latency/analyse.py) and its axis runs from -0.10
to 0.50 so that every cell is shown, with the legend above the data.  All text is 8 pt at
the 180-mm print width (figstyle.TEXT), so the canvas is 1440 x 570 px and panel a is
narrower to leave room for its end-of-line labels.
Run: /mmfs1/storage/users/xiar3/envs/figenv/bin/python scripts/fig_delay_curves.py
"""
import json
import math

import figstyle as F

S = json.loads((F.ROOT.parent / "results" / "latency" / "summary.json").read_text())
DS = [1, 2, 4, 8, 16]
P = lambda d, key: S["pooled"][f"d{d}"][key]
CELLS = [S["cells"][c]["arms"] for c in S["hypotheses"]["eligible_cells"]]
LOSS = {"F": lambda m, d: m["paused1"] - m[f"paused{d}"],
        "M_aligned": lambda m, d: m[f"paused{d}"] - m[f"aligned{d}"],
        "M_cpp": lambda m, d: m[f"paused{d}"] - m[f"cpp{d}"]}
cell_values = lambda d, key: [LOSS[key](m, d) for m in CELLS]
assert all(math.isclose(sum(cell_values(d, k)) / len(CELLS), P(d, k)["est"], abs_tol=1e-12) for d in DS for k in LOSS)
italic = lambda s: (s, {"italic": True})
W, H = 1440, 570
YT, YB = 100, 470                                       # plot top and bottom, both panels
body = []

# ---------------------------------------------------------------- panel a
X0, X1 = 100, 560
xs = [120 + 105 * i for i in range(len(DS))]          # log2-spaced delays
ylo, yhi = 0.1, 0.5
sy = (YB - YT - 10) / (yhi - ylo)
ya = lambda v: YB - (v - ylo) * sy
body.append(F.panel_title(30, 44, "a", "Return under delay"))
for k in range(5):
    v = ylo + 0.1 * k
    body.append(F.line(X0, ya(v), X1, ya(v), F.GRID))
    body.append(F.text(X0 - 14, ya(v), f"{v:.1f}", fill=F.MUTED, anchor="end", baseline="middle"))
body.append(F.line(X0, YB, X1, YB, F.AXIS, 1.5))
for x, d in zip(xs, DS):
    body.append(F.text(x, YB + 32, str(d), fill=F.MUTED, anchor="middle"))
body.append(F.text((X0 + X1) / 2, YB + 68, ["delay ", italic("d"), " (decisions)"], anchor="middle"))
body.append(F.text(34, (YT + YB) / 2, "normalised return", anchor="middle", rotate=-90))
p1 = P(1, "R_paused1")["est"]
body.append(F.line(X0, ya(p1), X1, ya(p1), F.GREY, 1.3, "5 4"))
body.append(F.text(X1, ya(p1) - 12, "paused, replan every decision", fill=F.MUTED, anchor="end"))
ARMS = [("R_paused_d", "paused, replan every d", F.NAVY, None),
        ("R_cpp", "committed prefix", F.GREEN, None),
        ("R_aligned", "time-aligned", F.ORANGE, None),
        ("R_naive", "naive", F.GREY, "9 6")]
for key, _, col, _ in ARMS:                             # bands first, under every line
    lo = [ya(P(d, key)["ci95"][0]) for d in DS]; hi = [ya(P(d, key)["ci95"][1]) for d in DS]
    pts = [f"{x} {y:.2f}" for x, y in zip(xs, hi)] + [f"{x} {y:.2f}" for x, y in zip(xs[::-1], lo[::-1])]
    body.append(f'<path d="M {" L ".join(pts)} Z" fill="{col}" fill-opacity="0.10"></path>')
ends = []
for key, lab, col, dash in ARMS:
    ys = [ya(P(d, key)["est"]) for d in DS]
    da = f' stroke-dasharray="{dash}"' if dash else ""
    body.append(f'<path d="M {" L ".join(f"{x} {y:.2f}" for x, y in zip(xs, ys))}" fill="none" stroke="{col}" '
                f'stroke-width="3"{da} stroke-linejoin="round"></path>')
    body += [f'<circle cx="{x}" cy="{y:.2f}" r="5.5" fill="{col}" stroke="#fff" stroke-width="2"></circle>'
             for x, y in zip(xs, ys)]
    ends.append([ys[-1], lab, col])
ends.sort()                                             # nudge end labels at least 27 px apart
for _ in range(20):
    for i in range(len(ends) - 1):
        gap = ends[i + 1][0] - ends[i][0]
        if gap < 27:
            ends[i][0] -= (27 - gap) / 2; ends[i + 1][0] += (27 - gap) / 2
for y, lab, col in ends:
    body.append(F.text(xs[-1] + 14, y, lab, fill=col, weight=600, baseline="middle"))

# ---------------------------------------------------------------- panel b
BX0, BX1 = 945, 1430
YLO, YHI = -0.10, 0.50                                  # covers every cell value, legend above the data
sb = (YB - YT) / (YHI - YLO)
yb = lambda v: YB - (v - YLO) * sb
BY0 = yb(0)
dot = lambda x, y: (f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.4" fill="#fff" stroke="{F.INK}" '
                    f'stroke-width="1.2"></circle>')
body.append(F.panel_title(842, 44, "b", "Where the delay loss comes from"))
for k in range(7):
    v = YLO + 0.1 * k
    body.append(F.line(BX0, yb(v), BX1, yb(v), F.GRID))
    body.append(F.text(BX0 - 14, yb(v), f"{v:.2f}".replace("-", "−"), fill=F.MUTED, anchor="end",
                       baseline="middle"))
centers = [BX0 + 48 + 97 * i for i in range(len(DS))]
BARS = [("F", "feedback-rate loss F(d)", F.NAVY, -40),
        ("M_aligned", "mismatch, time-aligned", F.ORANGE, -12),
        ("M_cpp", "mismatch, committed prefix", F.GREEN, 16)]
for j, (key, lab, col, off) in enumerate(BARS):
    for c, d in zip(centers, DS):
        r = P(d, key); x = c + off
        h = yb(0) - yb(r["est"])
        if abs(h) < 1.5 and r["ci95"][0] == r["ci95"][1]:   # exactly zero by construction (F at d = 1)
            body.append(F.line(x, BY0, x + 24, BY0, col, 3)); continue
        body.append(F.rect(x, min(yb(0), yb(r["est"])), 24, max(abs(h), 1.33), col, rx=2))
        # one dot per eligible cell, always in the same horizontal order, under the interval
        vals = cell_values(d, key)
        body += [dot(x + 12 + (i - (len(vals) - 1) / 2) * 2.4, yb(v)) for i, v in enumerate(vals)]
        lo, hi = yb(r["ci95"][0]), yb(r["ci95"][1])
        body += [F.line(x + 12, lo, x + 12, hi, F.BODY, 1.4), F.line(x + 7, lo, x + 17, lo, F.BODY, 1.4),
                 F.line(x + 7, hi, x + 17, hi, F.BODY, 1.4)]
    body.append(F.rect(955, YT + 4 + 30 * j, 24, 16, col, rx=2))
    body.append(F.text(989, YT + 12 + 30 * j, lab, baseline="middle"))
body.append(dot(967, YT + 102))
body.append(F.text(989, YT + 102, "family–task cell", baseline="middle"))
body.append(F.line(BX0, BY0, BX1, BY0, F.AXIS, 1.5))
for c, d in zip(centers, DS):
    body.append(F.text(c, YB + 32, str(d), fill=F.MUTED, anchor="middle"))
body.append(F.text((BX0 + BX1) / 2, YB + 68, ["delay ", italic("d"), " (decisions)"], anchor="middle"))
body.append(F.text(852, (YT + YB) / 2, "loss in normalised return", anchor="middle", rotate=-90))
F.render(F.svg(W, H, body), "delay_curves")

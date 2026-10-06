"""Figure 3: closed-loop return against delay (Study 1, pooled over eligible cells) and the
split of the delay loss into feedback-rate and mismatch terms (results/latency/summary.json).

Redesign of 6 October 2026 (Claude Design handoff, figures/delay_curves.svg):
1440 x 600 px; panel a with direct end-of-line labels and 95 % bands, panel b grouped bars
with 95 % intervals.
Run: /mmfs1/storage/users/xiar3/envs/figenv/bin/python scripts/fig_delay_curves.py
"""
import json

import figstyle as F

S = json.loads((F.ROOT.parent / "results" / "latency" / "summary.json").read_text())
DS = [1, 2, 4, 8, 16]
P = lambda d, key: S["pooled"][f"d{d}"][key]
italic = lambda s: (s, {"italic": True})
body = []

# ---------------------------------------------------------------- panel a
X0, X1, Y0 = 130, 590, 440
xs = [150 + 105 * i for i in range(len(DS))]          # log2-spaced delays
ylo, yhi = 0.1, 0.5
sy = (Y0 - 116.19) / (yhi - ylo)
ya = lambda v: Y0 - (v - ylo) * sy
body.append(F.panel_title(60, 52, "a", "Return under delay"))
for k in range(5):
    v = ylo + 0.1 * k
    body.append(F.line(X0, ya(v), X1, ya(v), F.GRID))
    body.append(F.text(X0 - 14, ya(v), f"{v:.1f}", size=19, fill=F.MUTED, anchor="end", baseline="middle"))
body.append(F.line(X0, Y0, X1, Y0, F.AXIS, 1.5))
for x, d in zip(xs, DS):
    body.append(F.text(x, 470, str(d), size=19, fill=F.MUTED, anchor="middle"))
body.append(F.text((X0 + X1) / 2, 506, ["delay ", italic("d"), " (decisions)"], size=21, anchor="middle"))
body.append(F.text(58, 270, "normalised return", size=21, anchor="middle", rotate=-90))
p1 = P(1, "R_paused1")["est"]
body.append(F.line(X0, ya(p1), X1, ya(p1), F.GREY, 1.3, "5 4"))
body.append(F.text(570, ya(p1) - 12, "paused, replan every decision", size=17, fill=F.MUTED, anchor="end"))
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
ends.sort()                                             # nudge end labels at least 24 px apart
for _ in range(20):
    for i in range(len(ends) - 1):
        gap = ends[i + 1][0] - ends[i][0]
        if gap < 24:
            ends[i][0] -= (24 - gap) / 2; ends[i + 1][0] += (24 - gap) / 2
for y, lab, col in ends:
    body.append(F.text(584, y, lab, size=18, fill=col, weight=600, baseline="middle"))

# ---------------------------------------------------------------- panel b
BX0, BX1, BY0 = 910, 1410, 395.65
sb = (BY0 - 100) / 0.20
yb = lambda v: BY0 - v * sb
body.append(F.panel_title(840, 52, "b", "Where the delay loss comes from"))
for k in range(5):
    v = 0.05 * k
    body.append(F.line(BX0, yb(v), BX1, yb(v), F.GRID))
    body.append(F.text(BX0 - 14, yb(v), f"{v:.2f}", size=19, fill=F.MUTED, anchor="end", baseline="middle"))
centers = [960 + 100 * i for i in range(len(DS))]
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
        lo, hi = yb(r["ci95"][0]), yb(r["ci95"][1])
        body += [F.line(x + 12, lo, x + 12, hi, F.BODY, 1.4), F.line(x + 7, lo, x + 17, lo, F.BODY, 1.4),
                 F.line(x + 7, hi, x + 17, hi, F.BODY, 1.4)]
    body.append(F.rect(920, 101 + 28 * j, 22, 14, col, rx=2))
    body.append(F.text(952, 108 + 28 * j, lab, size=18, baseline="middle"))
body.append(F.line(BX0, BY0, BX1, BY0, F.AXIS, 1.5))
for c, d in zip(centers, DS):
    body.append(F.text(c, 470, str(d), size=19, fill=F.MUTED, anchor="middle"))
body.append(F.text(1160, 506, ["delay ", italic("d"), " (decisions)"], size=21, anchor="middle"))
body.append(F.text(838, 270, "loss in normalised return", size=21, anchor="middle", rotate=-90))
F.render(F.svg(1440, 600, body), "delay_curves")

"""Figure 4: where hedging helps.  Gain over committed-prefix planning of RCH (K = 4) and of
hedge-then-refine, per family-task cell (mean of three models) and pooled over the 15 eligible cells
with 95 % hierarchical bootstrap intervals, at d = 4 (a) and d = 8 (b).
Data: results/latency/htr.json (Section 17A; the RCH and committed-prefix arms are those of Sections
12 and 16B).  All text is 8 pt at the 180-mm print width (figstyle.TEXT).
Run: /mmfs1/storage/users/xiar3/envs/figenv/bin/python scripts/fig_gains.py
"""
import json

import figstyle as F

R = json.loads((F.ROOT.parent / "results" / "latency" / "htr.json").read_text())
FAM = {"rssm": "RSSM", "jepa": "JEPA", "tdmpc2": "TD-MPC2"}
TASK = {"cartpole-swingup": "cartpole", "cup-catch": "cup catch", "walker-walk": "walker",
        "reacher-easy": "reacher", "cheetah-run": "cheetah", "finger-spin": "finger"}
ORDER = ["cartpole-swingup", "cup-catch", "walker-walk", "reacher-easy", "cheetah-run", "finger-spin"]
cells = sorted(R["per_cell"], key=lambda c: (ORDER.index(c.split("/")[1]), ["rssm", "jepa", "tdmpc2"].index(c.split("/")[0])))
W, ROW, Y0 = 1440, 31, 118
H = Y0 + ROW * (len(cells) + 2) + 92
XL = 24                                         # label column
PANELS = [(4, 330, 860, "a"), (8, 900, 1420, "b")]
LO, HI = -0.06, 0.18
body = []


def xs(v, x0, x1):
    return x0 + (v - LO) / (HI - LO) * (x1 - x0)


def gain(c, d, arm):
    v = R["per_cell"][c][f"d{d}"]
    return v[arm] - v[f"hcpp{d}k1"]


yb = Y0 + ROW * len(cells)
for i, c in enumerate(cells):                   # row labels and task separators
    y = Y0 + ROW * i + ROW / 2
    f, t = c.split("/")
    new = t in ("reacher-easy", "cheetah-run", "finger-spin")
    body.append(F.text(XL, y, f"{FAM[f]} {TASK[t]}" + ("*" if new else ""), fill=F.INK, baseline="middle"))
    if i and c.split("/")[1] != cells[i - 1].split("/")[1]:
        for _, x0, x1, _ in PANELS:
            body.append(F.line(x0, Y0 + ROW * i, x1, Y0 + ROW * i, F.GRID, 1))
body.append(F.text(XL, yb + ROW * 1.1, [("Pooled, 15 cells", {"weight": 600})], baseline="middle"))
for d, x0, x1, letter in PANELS:
    body.append(F.text(x0, Y0 - 34, [(letter, {"weight": 700}), (f"Delay d = {d}", {"weight": 600, "dx": 12})], size=F.TEXT))
    for v in (-0.05, 0.0, 0.05, 0.10, 0.15):
        x = xs(v, x0, x1)
        body.append(F.line(x, Y0 - 6, x, yb + ROW * 1.8, F.AXIS if v == 0 else F.GRID, 1.5 if v == 0 else 1))
        body.append(F.text(x, yb + ROW * 1.8 + 30, f"{v:+.2f}".replace("+0.00", "0"), fill=F.MUTED, anchor="middle"))
    for i, c in enumerate(cells):
        y = Y0 + ROW * i + ROW / 2
        r, h = gain(c, d, f"hcpp{d}k4"), gain(c, d, f"rcpp{d}k4r1")
        body.append(F.line(xs(r, x0, x1), y, xs(h, x0, x1), y, F.LIGHT, 3))
        body.append(f'<circle cx="{xs(r, x0, x1):.1f}" cy="{y:.1f}" r="7" fill="{F.GREEN}" stroke="#fff" stroke-width="1.5"></circle>')
        body.append(F.rect(xs(h, x0, x1) - 6.5, y - 6.5, 13, 13, F.NAVY, rx=1))
    y = yb + ROW * 1.1                           # pooled estimates with 95 % intervals
    for key, col, dy, shape in ((f"RCH_vs_CPP_d{d}", F.GREEN, -7, "c"), (f"HTR_vs_CPP_d{d}", F.NAVY, 7, "s")):
        v = R["secondary"][key]
        lo, hi = xs(v["ci95"][0], x0, x1), xs(v["ci95"][1], x0, x1)
        body += [F.line(lo, y + dy, hi, y + dy, col, 2.2), F.line(lo, y + dy - 6, lo, y + dy + 6, col, 2.2),
                 F.line(hi, y + dy - 6, hi, y + dy + 6, col, 2.2)]
        xe = xs(v["est"], x0, x1)
        body.append(f'<circle cx="{xe:.1f}" cy="{y + dy:.1f}" r="7" fill="{F.GREEN}" stroke="#fff" stroke-width="1.5"></circle>'
                    if shape == "c" else F.rect(xe - 6.5, y + dy - 6.5, 13, 13, F.NAVY, rx=1))
body.append(F.text((PANELS[0][1] + PANELS[1][2]) / 2, H - 22, "gain over committed-prefix planning (normalised return)",
                   anchor="middle"))
lx, ly = PANELS[0][1], 34                        # legend, one line above the panels
l1 = "residual-calibrated hedging (K = 4)"
body.append(f'<circle cx="{lx + 7}" cy="{ly}" r="7" fill="{F.GREEN}"></circle>')
body.append(F.text(lx + 22, ly, l1, baseline="middle"))
lx2 = lx + 22 + F.width([l1], F.TEXT) + 44
body.append(F.rect(lx2, ly - 6.5, 13, 13, F.NAVY, rx=1))
body.append(F.text(lx2 + 21, ly, "hedge-then-refine", baseline="middle"))
body.append(F.text(XL, H - 22, "* task added after the method was fixed", fill=F.MUTED))
F.render(F.svg(W, H, body), "hedge_gains")

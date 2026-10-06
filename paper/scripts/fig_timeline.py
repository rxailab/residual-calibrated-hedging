"""Figure 2: planning schedules under a delay of d decisions (paused, time-aligned,
committed prefix).  Schematic, no data.

Redesign of 6 October 2026 (Claude Design handoff, figures/cpp_timeline.svg):
1440 x 900 px; plans in navy, executed slices in light blue, the slices executed from a
state the plan did not assume outlined in orange, those planned for the predicted state
in green.
Run: /mmfs1/storage/users/xiar3/envs/figenv/bin/python scripts/fig_timeline.py
"""
import figstyle as F

D, N = 4, 4                       # delay in decisions, number of intervals drawn
X0, SLOT = 400, 250               # x of decision 0, px per interval of d decisions
BOX_W, PLAN_H, EXEC_H = 244, 58, 66
italic = lambda s: (s, {"italic": True})
sub = lambda s: (s, {"sub": True})


def label(y_top, title, sub1=None, sub2=None, title_weight=600):
    out = [F.text(340, y_top, title, size=21, anchor="end", weight=title_weight)]
    if sub1:
        out.append(F.text(340, y_top + 26, sub1, size=21, fill=F.MUTED if sub2 else F.INK, anchor="end"))
    if sub2:
        out.append(F.text(340, y_top + 52, sub2, size=21, fill=F.MUTED, anchor="end"))
    return out


def exec_box(x, y, segs, outline=None):
    kw = {"stroke": outline, "stroke_width": "3"} if outline else {}
    return [F.rect(x, y, BOX_W, EXEC_H, F.LIGHT, rx=3, **kw),
            F.text(x + BOX_W / 2, y + EXEC_H / 2, segs, size=19, anchor="middle", baseline="middle")]


def arrow(x, y):
    return f'<path d="M {x:.0f} {y:.0f} L {x + 16:.0f} {y + 22:.0f}" stroke="{F.MUTED}" stroke-width="1.6" fill="none" marker-end="url(#arr)"></path>'


body = [f'<defs><marker id="arr" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">'
        f'<path d="M0,0 L8,4 L0,8 z" fill="{F.MUTED}"></path></marker></defs>']

# row 1: paused simulator
body += label(103, "Paused simulator", "(frozen while planning;", "the standard report)")
body.append(F.line(384, 179, 1416, 179, F.GRID, 2))
for k in range(N):
    x = X0 + k * SLOT
    body.append(F.rect(x, 60, 26.25, PLAN_H, F.NAVY, rx=3))
    body.append(F.text(x + 36.25, 89, "plan", size=17, fill=F.NAVY, baseline="middle"))
    body += exec_box(x + 3, 146, [italic("A"), sub(str(k)), " [0:", italic("d"), "]"])

# rows 2 and 3: asynchronous schedules
for row, (y_plan, title, subtitle, predicted, colour) in enumerate([
        (290, "Asynchronous,", "time-aligned", False, F.ORANGE),
        (520, "Asynchronous,", "committed prefix", True, F.GREEN)]):
    y_exec = y_plan + 86
    body += label(y_plan + 76, title, subtitle)
    body.append(F.line(384, y_exec + EXEC_H / 2, 1416, y_exec + EXEC_H / 2, F.GRID, 2))
    for k in range(N):
        x = X0 + 3 + k * SLOT
        t = (k + 1) * D if predicted else k * D
        state = [("ŝ" if predicted else "s", {"italic": True}), sub(str(t))]
        body.append(F.rect(x, y_plan, BOX_W, PLAN_H, F.NAVY, rx=3))
        body.append(F.text(x + BOX_W / 2, y_plan + PLAN_H / 2, ["plan from "] + state, size=19, fill="#fff",
                           anchor="middle", baseline="middle"))
        if k == 0:
            body += exec_box(x, y_exec, ["bootstrap"])
        else:
            body.append(arrow(x - 13, y_plan + 60))
            body += exec_box(x, y_exec, [italic("A"), sub(str(k - 1)), " [", italic("d"), ":2", italic("d"), "]"],
                             outline=colour)

# decision axis
body.append(F.line(X0, 706, X0 + N * SLOT, 706, F.AXIS, 1.5))
for k in range(N + 1):
    x = X0 + k * SLOT
    body.append(F.line(x, 706, x, 714, F.AXIS, 1.5))
    body.append(F.text(x, 740, str(k * D), size=19, fill=F.MUTED, anchor="middle"))
body.append(F.text(X0 + N * SLOT / 2, 774, ["decision index  (delay ", italic("d"), f" = {D} decisions)"], size=21,
                   anchor="middle"))

# legend
body.append(F.rect(400, 856, 26, 16, F.LIGHT, rx=2, stroke=F.ORANGE, stroke_width="3"))
body.append(F.text(438, 864, ["plan made for ", italic("s"), sub("t"), ", executed from ", italic("s"), sub("t+d")],
                   size=18, fill=F.MUTED, baseline="middle"))
body.append(F.rect(850, 856, 26, 16, F.LIGHT, rx=2, stroke=F.GREEN, stroke_width="3"))
body.append(F.text(888, 864, ["plan made for the predicted ", ("ŝ", {"italic": True}), sub("t+d")],
                   size=18, fill=F.MUTED, baseline="middle"))
F.render(F.svg(1440, 900, body), "cpp_timeline")

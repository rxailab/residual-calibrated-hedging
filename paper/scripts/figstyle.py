"""Shared style and SVG helpers for the side-paper figures (redesign of 6 October 2026,
from the Claude Design handoff "MetricTrust figures redesign").

Figures are written as SVG in a 1440-px-wide coordinate space (180 mm double column),
then rendered with cairosvg to PDF (vector, for LaTeX) and PNG (2x).  Text uses Albert
Sans (OFL, sidepaper/fonts), found through sidepaper/fonts/fonts.conf.

cairosvg places centred or right-aligned text made of several styled <tspan> pieces
incorrectly, so `text()` measures each piece with the real font (Pillow) and writes
such text left-aligned at the computed start position.  Single-style text keeps its
text-anchor.  Run the figure scripts with /mmfs1/storage/users/xiar3/envs/figenv/bin/python.
"""
import os
import pathlib
from xml.sax.saxutils import escape

ROOT = pathlib.Path(__file__).resolve().parents[1]
FONTS = ROOT / "fonts"
os.environ.setdefault("FONTCONFIG_FILE", str(FONTS / "fonts.conf"))
import cairosvg  # noqa: E402  (after FONTCONFIG_FILE is set)
from PIL import ImageFont  # noqa: E402

FAMILY = "'Albert Sans','Helvetica Neue',Helvetica,Arial,sans-serif"
INK, MUTED, BODY = "#1a1d21", "#6b7280", "#374151"
GRID, AXIS = "#e5e7eb", "#9ca3af"
NAVY, ORANGE, GREEN, GREY = "#0f3b5f", "#e07a2f", "#2f8f6f", "#7a8290"
LIGHT, ORANGE_TEXT = "#c9d7e3", "#9a4f17"
WEIGHTS = {400: "Regular", 500: "Medium", 600: "SemiBold", 700: "Bold"}
_FONT_CACHE = {}


def _font(size, weight=400, italic=False):
    key = (size, weight, italic)
    if key not in _FONT_CACHE:
        f = ImageFont.truetype(str(FONTS / ("AlbertSans-Italic[wght].ttf" if italic else "AlbertSans[wght].ttf")), size)
        name = WEIGHTS[weight] if not italic else ("Italic" if weight == 400 else WEIGHTS[weight] + " Italic")
        f.set_variation_by_name(name)
        _FONT_CACHE[key] = f
    return _FONT_CACHE[key]


def width(segments, size, weight=400):
    """Width in px of a list of segments (see `text`)."""
    w = 0.0
    for seg in segments:
        s, opt = (seg, {}) if isinstance(seg, str) else seg
        w += opt.get("dx", 0) + _font(opt.get("size", size), opt.get("weight", weight), opt.get("italic", False)).getlength(s)
    return w


def text(x, y, segments, size=19, fill=INK, anchor="start", weight=400, baseline=None, rotate=None):
    """SVG <text>.  `segments` is a string or a list of str / (str, opts) pieces; opts:
    italic, weight, size, sub (subscript: smaller and lowered), dx, fill."""
    if isinstance(segments, str):
        segments = [segments]
    multi = len(segments) > 1 or not isinstance(segments[0], str)
    if multi and anchor != "start":
        w = width(_norm(segments, size), size, weight)
        x = x - (w / 2 if anchor == "middle" else w)
        anchor = "start"
    attrs = [f'x="{x:.2f}"', f'y="{y:.2f}"', f'font-size="{size}"', f'fill="{fill}"']
    if anchor != "start":
        attrs.append(f'text-anchor="{anchor}"')
    if weight != 400:
        attrs.append(f'font-weight="{weight}"')
    if baseline:
        attrs.append(f'dominant-baseline="{baseline}"')
    if rotate is not None:
        attrs.append(f'transform="rotate({rotate} {x:.2f} {y:.2f})"')
    body, lowered = [], False
    for seg in segments:
        s, o = (seg, {}) if isinstance(seg, str) else seg
        ta = []
        if o.get("italic"):
            ta.append('font-style="italic"')
        if o.get("weight"):
            ta.append(f'font-weight="{o["weight"]}"')
        if o.get("fill"):
            ta.append(f'fill="{o["fill"]}"')
        if o.get("dx"):
            ta.append(f'dx="{o["dx"]}"')
        if o.get("sub"):
            ta += [f'font-size="{o.get("size", 14)}"', 'dy="6"' if not lowered else ""]
            lowered = True
        elif lowered:
            ta.append('dy="-6"'); lowered = False
        if o.get("size") and not o.get("sub"):
            ta.append(f'font-size="{o["size"]}"')
        body.append(f'<tspan {" ".join(t for t in ta if t)}>{escape(s)}</tspan>' if ta else escape(s))
    return f'<text {" ".join(attrs)}>{"".join(body)}</text>'


def _norm(segments, size):
    out = []
    for seg in segments:
        s, o = (seg, {}) if isinstance(seg, str) else seg
        o = dict(o)
        if o.get("sub"):
            o.setdefault("size", 14)
        out.append((s, o))
    return out


def rect(x, y, w, h, fill, rx=0, **kw):
    extra = "".join(f' {k.replace("_", "-")}="{v}"' for k, v in kw.items())
    return f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" fill="{fill}"' + \
           (f' rx="{rx}"' if rx else "") + extra + "></rect>"


def line(x1, y1, x2, y2, stroke, width=1, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{stroke}" stroke-width="{width}"{d}></line>'


def panel_title(x, y, letter, title):
    return text(x, y, [(letter, {"weight": 700}), (title, {"weight": 600, "dx": 14})], size=25)


def svg(w, h, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
            f'font-family="{FAMILY}">\n<rect width="{w}" height="{h}" fill="#fff"></rect>\n' + "".join(body) + "\n</svg>\n")


def render(svg_text, name):
    """Write figures/<name>.svg, .pdf (vector) and .png (2x)."""
    out = ROOT / "figures"
    (out / f"{name}.svg").write_text(svg_text)
    cairosvg.svg2pdf(bytestring=svg_text.encode(), write_to=str(out / f"{name}.pdf"))
    w = int(svg_text.split('width="', 1)[1].split('"', 1)[0])
    cairosvg.svg2png(bytestring=svg_text.encode(), write_to=str(out / f"{name}.png"), output_width=2 * w)
    print("wrote", out / f"{name}.svg", "+ .pdf + .png")

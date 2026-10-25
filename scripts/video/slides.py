"""Slide cards (1920x1080 PNG) in the style of the post figures: background #fcfcfb, ink #0b0b0b.

Text is set in DejaVu Sans (shipped with matplotlib, the font of the post figures). The bottom band
(y >= BAND_Y) is kept free for burned-in captions.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
BG, INK, INK2, RULE, ORANGE, PURPLE = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#eb6834", "#4a3aa7"
BAND_Y = 880          # captions live below this line
MARGIN_X = 140

_FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
FONT_DIR = _FONT_DIR


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(_FONT_DIR / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")), size)


def _wrap(draw, text, fnt, max_w):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=fnt) <= max_w:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _text(draw, xy, text, fnt, fill=INK, max_w=None, gap=1.28):
    """Draw wrapped text, return the y below it."""
    x, y = xy
    lines = _wrap(draw, text, fnt, max_w) if max_w else [text]
    lh = int(fnt.size * gap)
    for ln in lines:
        draw.text((x, y), ln, font=fnt, fill=fill)
        y += lh
    return y


def _canvas():
    im = Image.new("RGB", (W, H), BG)
    return im, ImageDraw.Draw(im)


def _accent(d, y=96):
    d.rectangle([MARGIN_X, y, MARGIN_X + 96, y + 8], fill=ORANGE)


def title_card(c):
    im, d = _canvas()
    _accent(d, 250)
    y = _text(d, (MARGIN_X, 300), c["title"], font(84, True), INK, W - 2 * MARGIN_X, 1.18)
    y = _text(d, (MARGIN_X, y + 26), c["subtitle"], font(40), INK2, W - 2 * MARGIN_X)
    d.line([MARGIN_X, 830, W - MARGIN_X, 830], fill=RULE, width=2)
    d.text((MARGIN_X, 850), c["byline"], font=font(32), fill=INK)
    return [im]


def _table(d, x, y, header, rows, widths, fnt_h, fnt_b, row_h, wrap=False):
    """Plain table: bold header, thin rules, wrapped cells. Returns the y below."""
    cx = [x]
    for w in widths[:-1]:
        cx.append(cx[-1] + w)
    for i, h in enumerate(header):
        d.text((cx[i], y), h, font=fnt_h, fill=INK2)
    y += int(fnt_h.size * 1.5)
    d.line([x, y, x + sum(widths), y], fill=INK2, width=2)
    y += 14
    for r in rows:
        nl = 1
        for i, cell in enumerate(r):
            lines = _wrap(d, cell, fnt_b, widths[i] - 28) if wrap else [cell]
            nl = max(nl, len(lines))
            yy = y
            for ln in lines:
                d.text((cx[i], yy), ln, font=fnt_b, fill=(ORANGE if i == len(r) - 1 and not wrap else INK), )
                yy += int(fnt_b.size * 1.3)
        y += max(row_h, int(nl * fnt_b.size * 1.3) + 16)
        d.line([x, y - 8, x + sum(widths), y - 8], fill=RULE, width=2)
    return y


def text_card(c):
    im, d = _canvas()
    _accent(d, c.get("y0", 124) - 28)
    y = _text(d, (MARGIN_X, c.get("y0", 124)), c["heading"], font(60, True), INK, W - 2 * MARGIN_X)
    y += 18
    if c.get("lead"):
        y = _text(d, (MARGIN_X, y + 10), c["lead"], font(44), INK2, W - 2 * MARGIN_X)
        y += 24
    if c.get("stats"):
        x = MARGIN_X
        for s in c["stats"]:
            fb, fs = font(78, True), font(32)
            wbig = d.textlength(s["big"], font=fb)
            wsm = d.textlength(s["small"], font=fs)
            bw = int(max(wbig, wsm)) + 70
            d.rounded_rectangle([x, y + 10, x + bw, y + 200], radius=18, outline=RULE, width=3, fill="#ffffff")
            d.text((x + 35, y + 30), s["big"], font=fb, fill=ORANGE)
            d.text((x + 35, y + 140), s["small"], font=fs, fill=INK2)
            x += bw + 40
        y += 240
    for b in c.get("bullets", []):
        d.ellipse([MARGIN_X + 2, y + 20, MARGIN_X + 18, y + 36], fill=ORANGE)
        y = _text(d, (MARGIN_X + 44, y + 6), b, font(40), INK, W - 2 * MARGIN_X - 44, 1.25) + 18
    t = c.get("table")
    if t:
        if t.get("title"):
            d.text((MARGIN_X, y + 10), t["title"], font=font(34, True), fill=INK)
            y += 70
        else:
            y += 20
        if c.get("wide") or t.get("wide"):
            widths = [620, 1020]
            _table(d, MARGIN_X, y, t["header"], t["rows"], widths, font(32, True), font(34), 100, wrap=True)
        else:
            widths = [520, 400, 400, 380]
            y = _table(d, MARGIN_X, y, t["header"], t["rows"], widths, font(32, True), font(36), 70)
    if c.get("footnote"):
        d.text((MARGIN_X, BAND_Y - 70), c["footnote"], font=font(30), fill=INK2)
    return [im]


def figure_card(c, root: Path):
    img = Image.open(root / c["image"]).convert("RGB")
    out = []
    for v in c.get("views") or [{"at": 0}]:
        im, d = _canvas()
        d.text((MARGIN_X - 60, 36), c["heading"], font=font(46, True), fill=INK)
        d.line([MARGIN_X - 60, 108, W - MARGIN_X + 60, 108], fill=RULE, width=2)
        crop = img.crop(tuple(v["crop"])) if v.get("crop") else img
        box_w, box_h = W - 2 * 100, BAND_Y - 130 - 6
        s = min(box_w / crop.width, box_h / crop.height)
        fig = crop.resize((int(crop.width * s), int(crop.height * s)), Image.LANCZOS)
        im.paste(fig, ((W - fig.width) // 2, 128 + (box_h - fig.height) // 2))
        out.append(im)
    return out


def end_card(c, links: str):
    im, d = _canvas()
    _accent(d, 220)
    y = _text(d, (MARGIN_X, 270), c["title"], font(60, True), INK, W - 2 * MARGIN_X)
    y = _text(d, (MARGIN_X, y + 30), links, font(44), ORANGE, W - 2 * MARGIN_X, 1.3)
    d.line([MARGIN_X, 790, W - MARGIN_X, 790], fill=RULE, width=2)
    d.text((MARGIN_X, 815), c["author"], font=font(42, True), fill=INK)
    d.text((MARGIN_X, 880), c["contact"], font=font(36), fill=INK2)
    return [im]


def render(card: dict, root: Path, out_prefix: Path, links: str = "") -> list[Path]:
    kind = card["kind"]
    ims = {"title": lambda: title_card(card), "text": lambda: text_card(card),
           "figure": lambda: figure_card(card, root), "end": lambda: end_card(card, links)}[kind]()
    paths = []
    for i, im in enumerate(ims):
        p = out_prefix.with_name(f"{out_prefix.name}_v{i}.png")
        im.save(p)
        paths.append(p)
    return paths

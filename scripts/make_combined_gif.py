"""Build a single combined GIF for ALL environments: assets/gifs/all_envs.gif.

Environments are laid out side by side (horizontal). Each block is that
environment's clients (rows) against the three methods (columns), with an env
title band and method column headers; one shared client-label column sits on
the left and rows align across envs by client index. Source clips differ in
length within and across environments; every cell is mapped onto a shared
N-frame output timeline (start->end) and holds its last frame once it ends, so
all envs stay phase-aligned.

Cells are small and frames are subsampled to keep one big file manageable.
"""
import json
import os

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEL = os.path.join(ROOT, "assets", "video_selection.json")
OUT = os.path.join(ROOT, "assets", "gifs", "all_envs.gif")

METHODS = [("fedguide", "FedGuide"), ("fedguide_a", "FedGuide-A"), ("fedguide_p", "FedGuide-P")]

CELL = 72           # cell width/height in px
HEADER = 20         # method column-header band (per env block)
TITLEH = 26         # environment title band
LABELW = 46         # shared client-label column width
ENVGAP = 12         # horizontal gap between environment blocks
FPS = 12
N = 64              # number of output frames (shared timeline)
PAD = 2
BG = (15, 17, 21)
FG = (230, 230, 230)
MUTED = (154, 160, 170)
TITLEBG = (26, 29, 36)


def load_font(size):
    for p in ["/System/Library/Fonts/Helvetica.ttc",
              "/System/Library/Fonts/Supplemental/Arial.ttf"]:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


FONT = load_font(14)
FONT_SM = load_font(11)
FONT_TITLE = load_font(15)


def read_frames_at(path, wanted):
    """Read mp4, return {idx: CELLxCELL frame} for idx in `wanted`, plus length."""
    out = {}
    rd = imageio.get_reader(path)
    n = 0
    for i, fr in enumerate(rd):
        n = i + 1
        if i in wanted:
            out[i] = np.asarray(Image.fromarray(fr).resize((CELL, CELL), Image.LANCZOS))
    rd.close()
    return out, n


def text_centered(draw, cx, cy, s, font, fill):
    l, t, r, b = draw.textbbox((0, 0), s, font=font)
    draw.text((cx - (r - l) / 2, cy - (b - t) / 2), s, font=font, fill=fill)


def main():
    sel = json.load(open(SEL))
    envs = sel["environments"]

    # First pass: per env, find max clip length, then the source index each of
    # the N output frames maps to; gather the frames each cell needs.
    plan = []   # list of dicts per env
    for env, ed in envs.items():
        clients = sorted(ed["clients"], key=lambda x: int(x))
        # discover max length by scanning lengths cheaply via a full read of one
        # representative is unsafe (lengths vary), so read all and cache wanted.
        # Two sub-passes: (a) lengths, (b) wanted frames — merged into one read by
        # first guessing wanted from a provisional max, then correcting holds.
        lengths = {}
        # provisional read to get lengths AND keep all-needed frames later; to keep
        # it single-pass we read once with a broad wanted set (every index up to a
        # cap is impossible) -> instead do length pass via reader frame count.
        for client in clients:
            for key, _ in METHODS:
                path = ed["clients"][client][key]["video"]
                rd = imageio.get_reader(path)
                lengths[(client, key)] = rd.count_frames()
                rd.close()
        maxlen = max(lengths.values())
        # output frame k -> source index (round over this env's own max length)
        src_for_k = [int(round(k / (N - 1) * (maxlen - 1))) for k in range(N)]
        cells = {}
        for client in clients:
            for key, _ in METHODS:
                path = ed["clients"][client][key]["video"]
                clen = lengths[(client, key)]
                wanted = {min(s, clen - 1) for s in src_for_k}
                frames, _ = read_frames_at(path, wanted)
                cells[(client, key)] = frames
        plan.append(dict(env=env, ed=ed, clients=clients, lengths=lengths,
                         maxlen=maxlen, src_for_k=src_for_k, cells=cells))
        print("read", env, "(%d clients, maxlen %d)" % (len(clients), maxlen))

    # Geometry: env blocks side by side (horizontal), one shared client-label
    # column on the left. Rows align across envs by client index.
    methods_w = len(METHODS) * CELL + (len(METHODS) - 1) * PAD
    maxclients = max(len(pl["clients"]) for pl in plan)
    for i, pl in enumerate(plan):
        pl["x0"] = LABELW + PAD + i * (methods_w + ENVGAP)   # left edge of this env's cells
    total_w = LABELW + PAD + len(plan) * methods_w + (len(plan) - 1) * ENVGAP + PAD
    rows_top = TITLEH + HEADER + PAD
    total_h = rows_top + maxclients * (CELL + PAD) + PAD

    # Static base (titles, method headers, shared client labels) drawn once.
    base = Image.new("RGB", (total_w, total_h), BG)
    d = ImageDraw.Draw(base)
    # Shared client-label column.
    for r in range(maxclients):
        cy = rows_top + r * (CELL + PAD) + CELL / 2
        text_centered(d, LABELW / 2, cy, "c" + str(r), FONT_SM, MUTED)
    for pl in plan:
        x0 = pl["x0"]
        d.rectangle([x0 - PAD, 0, x0 + methods_w, TITLEH], fill=TITLEBG)
        text_centered(d, x0 + methods_w / 2, TITLEH / 2, pl["ed"]["display"], FONT_TITLE, FG)
        for c, (_, label) in enumerate(METHODS):
            cx = x0 + c * (CELL + PAD) + CELL / 2
            text_centered(d, cx, TITLEH + HEADER / 2, label, FONT_SM, FG)

    out_frames = []
    for k in range(N):
        canvas = base.copy()
        for pl in plan:
            x0 = pl["x0"]
            s = pl["src_for_k"][k]
            for r, client in enumerate(pl["clients"]):
                for c, (key, _) in enumerate(METHODS):
                    cell = pl["cells"][(client, key)]
                    clen = pl["lengths"][(client, key)]
                    fi = min(s, clen - 1)
                    img = cell.get(fi)
                    if img is None:  # nearest available cached frame
                        img = cell[min(cell, key=lambda j: abs(j - fi))]
                    x = x0 + c * (CELL + PAD)
                    y = rows_top + r * (CELL + PAD)
                    canvas.paste(Image.fromarray(img), (x, y))
        out_frames.append(np.asarray(canvas))

    imageio.mimsave(OUT, out_frames, fps=FPS, loop=0)
    print("wrote %s  (%dx%d, %d frames, %.1f MB)"
          % (OUT, total_w, total_h, N, os.path.getsize(OUT) / 1e6))


if __name__ == "__main__":
    main()

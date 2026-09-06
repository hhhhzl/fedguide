"""Build the README hero GIF from round-100 evaluation rollouts.

Default layout (``--layout clients``) is an environment x client grid for a
single algorithm: rows are environments, columns are federated clients. Because
every client of an environment runs a different dynamics/task variant, one row
reads as "the same policy family under heterogeneity" -- which is the claim the
paper makes. ``--layout methods`` reproduces the older per-environment
FedGuide / FedGuide-A / FedGuide-P comparison grid.

Source clips have very different lengths (Reacher ~51 frames, HalfCheetah
~1001). Each environment is mapped onto one shared N-frame output timeline over
its first ``WINDOW`` source frames, so every cell stays phase-aligned and short
clips hold their last frame instead of looping out of sync.

Encoding goes through the ffmpeg bundled with imageio-ffmpeg using
palettegen/paletteuse, which is dramatically smaller than a naive per-frame
palette (~4x on this content).

    python scripts/make_combined_gif.py                  # hero GIF
    python scripts/make_combined_gif.py --preview        # single PNG, fast
    python scripts/make_combined_gif.py --layout methods # variant comparison
"""
import argparse
import json
import os
import shutil
import subprocess
import tempfile

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEL = os.path.join(ROOT, "assets", "video_selection.json")
GIFDIR = os.path.join(ROOT, "assets", "gifs")

METHODS = [("fedguide", "FedGuide"), ("fedguide_a", "FedGuide-A"), ("fedguide_p", "FedGuide-P")]

# ---------------------------------------------------------------- appearance
BG = (255, 255, 255)
TILE_BG = (244, 245, 247)      # shows through the rounded corners
BORDER = (226, 229, 233)
INK = (17, 20, 26)             # environment names
MUTED = (140, 147, 158)        # column numbers, footnote
RADIUS = 6

CELL = 116
GAP = 7
MARGIN = 18
LABELW = 112                   # left column holding environment names

N = 64                         # output frames
FPS = 12.5
WINDOW = 240                   # max source frames consumed per env (~8s @30fps)
MAX_CLIENTS = 8                # methods layout only: clients are rows, so cap keeps it rectangular

# Square crop per environment as (center_x, center_y, side), all as fractions of
# the 480x480 source. Trims the black band above Reacher, the flat sky above the
# locomotion envs, and the letterbox bars beside MetaWorld.
CROPS = {
    "reacher": (0.50, 0.57, 0.86),
    "hopper": (0.50, 0.60, 0.84),
    "walker": (0.52, 0.60, 0.84),
    "halfcheetah": (0.52, 0.66, 0.76),
    "metaworld": (0.50, 0.545, 0.73),
}


def load_font(size, bold=False):
    candidates = [
        ("/System/Library/Fonts/HelveticaNeue.ttc", 1 if bold else 0),
        ("/System/Library/Fonts/Helvetica.ttc", 1 if bold else 0),
        ("/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold
         else "/System/Library/Fonts/Supplemental/Arial.ttf", 0),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
         else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 0),
    ]
    for path, index in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size, index=index)
            except Exception:
                continue
    return ImageFont.load_default()


def text_at(draw, x, y, s, font, fill, anchor="mm"):
    draw.text((x, y), s, font=font, fill=fill, anchor=anchor)


def rounded_mask(size, radius):
    w, h = size
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, w - 1, h - 1], radius=radius, fill=255)
    return m


# ------------------------------------------------------------------- decoding
def crop_box(env, w, h, aspect=1.0):
    """Crop region of the given tile aspect (w/h), centred per CROPS.

    `side` is tuned against the horizontal content extent (that is where the
    letterboxing is), so it always sets the crop width and the height follows
    from the aspect -- a narrower tile therefore keeps the full scene width and
    simply shows more of it vertically, rather than slicing the sides off.
    """
    cx, cy, side = CROPS.get(env, (0.5, 0.5, 1.0))
    cw = side * min(w, h)
    ch = cw / aspect
    if ch > h:
        cw, ch = cw * h / ch, h
    x0 = min(max(cx * w - cw / 2, 0), w - cw)
    y0 = min(max(cy * h - ch / 2, 0), h - ch)
    return (int(x0), int(y0), int(x0 + cw), int(y0 + ch))


def read_tiles(path, wanted, env, size):
    """Decode only `wanted` frame indices, cropped and resized to `size` (w, h)."""
    out = {}
    rd = imageio.get_reader(path)
    box = None
    for i, fr in enumerate(rd):
        if i in wanted:
            im = Image.fromarray(fr)
            if box is None:
                box = crop_box(env, *im.size, aspect=size[0] / size[1])
            out[i] = np.asarray(im.crop(box).resize(size, Image.LANCZOS))
        if wanted and i > max(wanted):
            break
    rd.close()
    return out


def clip_len(path):
    rd = imageio.get_reader(path)
    n = rd.count_frames()
    rd.close()
    return n


def timeline(win, nframes):
    """Output frame k -> source frame index, over `win` source frames."""
    return [int(round(k / (nframes - 1) * (win - 1))) for k in range(nframes)]


def collect(env, ed, clients, keys, size, nframes, preview):
    """Decode every (client, method) cell of one env onto the shared timeline."""
    lengths, paths = {}, {}
    for c in clients:
        for k in keys:
            p = ed["clients"][c][k]["video"]
            paths[(c, k)] = p if os.path.isabs(p) else os.path.join(ROOT, p)
            lengths[(c, k)] = clip_len(paths[(c, k)])
    win = min(max(lengths.values()), WINDOW)
    src = timeline(win, nframes)
    cells = {}
    for c in clients:
        for k in keys:
            n = lengths[(c, k)]
            wanted = {min(s, n - 1) for s in (src[len(src) // 2:len(src) // 2 + 1] if preview else src)}
            cells[(c, k)] = read_tiles(paths[(c, k)], wanted, env, size)
    return dict(env=env, ed=ed, clients=clients, lengths=lengths, src=src, cells=cells)


def pick(block, c, k, out_frame):
    """Tile for output frame index, holding the last frame of short clips."""
    cache = block["cells"][(c, k)]
    want = min(block["src"][out_frame], block["lengths"][(c, k)] - 1)
    hit = cache.get(want)
    if hit is None:
        hit = cache[min(cache, key=lambda j: abs(j - want))]
    return hit


# ------------------------------------------------------------------ encoding
def encode(frames, out, fps, preview):
    if preview:
        Image.fromarray(frames[0]).save(out)
        print("wrote %s  (%dx%d preview)" % (out, frames[0].shape[1], frames[0].shape[0]))
        return
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    tmp = tempfile.mkdtemp(prefix="fedguide_gif_")
    try:
        for i, f in enumerate(frames):
            Image.fromarray(f).save(os.path.join(tmp, "f%04d.png" % i))
        vf = ("[0:v] split [a][b];"
              "[a] palettegen=max_colors=224:stats_mode=diff [p];"
              "[b][p] paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle")
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-framerate", str(fps),
             "-i", os.path.join(tmp, "f%04d.png"), "-filter_complex", vf,
             "-loop", "0", out],
            check=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    h, w = frames[0].shape[:2]
    print("wrote %s  (%dx%d, %d frames, %.1f MB)"
          % (out, w, h, len(frames), os.path.getsize(out) / 1e6))


# ------------------------------------------------- layout 1: env x client grid
def row_geometry(n, row_w, gap):
    """Fit n tiles flush into row_w, spreading the rounding slack across the gaps.

    Environments do not all have the same client count (MetaWorld10 has 10, the
    rest have 8), so tile *width* is per row while row height stays CELL for
    every row. Every row therefore starts and ends on the same x and occupies the
    same height, which is what keeps the figure reading as one block.
    """
    cw = (row_w - (n - 1) * gap) // n
    extra = row_w - (n * cw + (n - 1) * gap)
    gaps = [gap + (1 if i < extra else 0) for i in range(n - 1)]
    xs, x = [], 0
    for i in range(n):
        xs.append(x)
        if i < n - 1:
            x += cw + gaps[i]
    return cw, xs


def render_clients(sel, algo, out, nframes, preview):
    envs = sel["environments"]
    label = dict(METHODS)[algo]

    # Geometry first: the sparsest row sets the square CELL, denser rows keep the
    # same height and narrow their tiles to land on the same right edge.
    rows = [(env, ed, sorted(ed["clients"], key=int)) for env, ed in envs.items()]
    ref = min(len(c) for _, _, c in rows)
    row_w = ref * CELL + (ref - 1) * GAP
    geom = [row_geometry(len(c), row_w, GAP) for _, _, c in rows]

    blocks = []
    for (env, ed, clients), (cw, _) in zip(rows, geom):
        blocks.append(collect(env, ed, clients, [algo], (cw, CELL), nframes, preview))
        print("read %-12s %d clients @ %dx%dpx" % (env, len(clients), cw, CELL))

    W = MARGIN + LABELW + row_w + MARGIN
    H = MARGIN + len(rows) * CELL + (len(rows) - 1) * GAP + MARGIN + 22
    x0, y0 = MARGIN + LABELW, MARGIN

    f_env, f_num, f_cnt = load_font(15, bold=True), load_font(11), load_font(10)
    base = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(base)
    for r, (b, (cw, xs)) in enumerate(zip(blocks, geom)):
        y = y0 + r * (CELL + GAP)
        text_at(d, x0 - 14, y + CELL / 2 - 7, b["ed"]["display"], f_env, INK, anchor="rm")
        text_at(d, x0 - 14, y + CELL / 2 + 9,
                "%d clients" % len(b["clients"]), f_cnt, MUTED, anchor="rm")
        for x in xs:
            d.rounded_rectangle([x0 + x, y, x0 + x + cw - 1, y + CELL - 1],
                                radius=RADIUS, fill=TILE_BG)
    text_at(d, x0, H - MARGIN - 4,
            "Round-100 evaluation rollouts  ·  %s  ·  each column is a different "
            "client with its own dynamics" % label, f_num, MUTED, anchor="lm")

    masks = {cw: rounded_mask((cw, CELL), RADIUS) for cw, _ in geom}
    frames = []
    for k in range(1 if preview else nframes):
        canvas = base.copy()
        dd = ImageDraw.Draw(canvas)
        for r, (b, (cw, xs)) in enumerate(zip(blocks, geom)):
            y = y0 + r * (CELL + GAP)
            for client, x in zip(b["clients"], xs):
                canvas.paste(Image.fromarray(pick(b, client, algo, k)),
                             (x0 + x, y), masks[cw])
                dd.rounded_rectangle([x0 + x, y, x0 + x + cw - 1, y + CELL - 1],
                                     radius=RADIUS, outline=BORDER, width=1)
        frames.append(np.asarray(canvas))
    encode(frames, out, FPS, preview)


# ------------------------------------------- layout 2: per-env method comparison
def render_methods(sel, out, nframes, preview):
    cell, gap = 78, 5
    envgap, titleh, headerh, labelw = 16, 26, 20, 52
    blocks = []
    for env, ed in sel["environments"].items():
        clients = sorted(ed["clients"], key=int)[:MAX_CLIENTS]
        blocks.append(collect(env, ed, clients, [k for k, _ in METHODS], (cell, cell), nframes, preview))
        print("read %-12s %d clients" % (env, len(clients)))

    mw = len(METHODS) * cell + (len(METHODS) - 1) * gap
    nrow = max(len(b["clients"]) for b in blocks)
    for i, b in enumerate(blocks):
        b["x0"] = MARGIN + labelw + i * (mw + envgap)
    W = MARGIN + labelw + len(blocks) * mw + (len(blocks) - 1) * envgap + MARGIN
    top = MARGIN + titleh + headerh
    H = top + nrow * (cell + gap) - gap + MARGIN

    f_title, f_sm, f_hdr = load_font(14, bold=True), load_font(10), load_font(10, bold=True)
    base = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(base)
    for r in range(nrow):
        text_at(d, MARGIN + labelw - 12, top + r * (cell + gap) + cell / 2,
                "Client %d" % (r + 1), f_sm, MUTED, anchor="rm")
    for b in blocks:
        bx = b["x0"]
        d.rounded_rectangle([bx, MARGIN, bx + mw - 1, MARGIN + titleh - 6],
                            radius=4, fill=(243, 244, 246))
        text_at(d, bx + mw / 2, MARGIN + (titleh - 6) / 2, b["ed"]["display"], f_title, INK)
        for c, (_, name) in enumerate(METHODS):
            text_at(d, bx + c * (cell + gap) + cell / 2, MARGIN + titleh + headerh / 2 - 4,
                    name, f_hdr, MUTED)
        for r in range(nrow):
            for c in range(len(METHODS)):
                x, y = bx + c * (cell + gap), top + r * (cell + gap)
                d.rounded_rectangle([x, y, x + cell - 1, y + cell - 1], radius=5, fill=TILE_BG)

    mask = rounded_mask((cell, cell), 5)
    frames = []
    for k in range(1 if preview else nframes):
        canvas = base.copy()
        dd = ImageDraw.Draw(canvas)
        for b in blocks:
            for r, client in enumerate(b["clients"]):
                for c, (key, _) in enumerate(METHODS):
                    x, y = b["x0"] + c * (cell + gap), top + r * (cell + gap)
                    canvas.paste(Image.fromarray(pick(b, client, key, k)), (x, y), mask)
                    dd.rounded_rectangle([x, y, x + cell - 1, y + cell - 1],
                                         radius=5, outline=BORDER, width=1)
        frames.append(np.asarray(canvas))
    encode(frames, out, FPS, preview)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", choices=["clients", "methods"], default="clients")
    ap.add_argument("--algo", default="fedguide", choices=[k for k, _ in METHODS])
    ap.add_argument("--frames", type=int, default=N)
    ap.add_argument("--preview", action="store_true", help="render one PNG instead of the GIF")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    sel = json.load(open(SEL))
    os.makedirs(GIFDIR, exist_ok=True)
    stem = "all_envs" if a.layout == "clients" else "all_envs_variants"
    out = a.out or os.path.join(GIFDIR, stem + (".png" if a.preview else ".gif"))
    if a.layout == "clients":
        render_clients(sel, a.algo, out, a.frames, a.preview)
    else:
        render_methods(sel, out, a.frames, a.preview)


if __name__ == "__main__":
    main()

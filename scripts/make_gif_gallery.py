"""Generate assets/gifs/index.html — a comparison gallery of policy rollout GIFs.

Reads assets/video_selection.json for display names, seeds and scores, and lays
out one section per environment with methods as columns and clients as rows.
GIF paths are relative to assets/gifs/ (env/method/client_<n>.gif).
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEL = os.path.join(ROOT, "assets", "video_selection.json")
OUT = os.path.join(ROOT, "assets", "gifs", "index.html")

METHODS = [("fedguide", "FedGuide"), ("fedguide_a", "FedGuide-A"), ("fedguide_p", "FedGuide-P")]


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


CSS = """
  :root { --bg:#0f1115; --card:#1a1d24; --fg:#e6e6e6; --muted:#9aa0aa; --accent:#6ea8fe; --border:#2a2e37; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }
  header { padding:28px 24px 8px; }
  h1 { margin:0 0 4px; font-size:24px; }
  .sub { color:var(--muted); font-size:14px; }
  nav { position:sticky; top:0; background:rgba(15,17,21,.92); backdrop-filter:blur(6px);
        padding:12px 24px; border-bottom:1px solid var(--border); z-index:10; display:flex; gap:10px; flex-wrap:wrap; }
  nav a { color:var(--accent); text-decoration:none; font-size:14px; padding:4px 10px;
          border:1px solid var(--border); border-radius:999px; }
  nav a:hover { background:var(--card); }
  section { padding:24px; }
  h2 { font-size:20px; margin:0 0 4px; }
  .envmeta { color:var(--muted); font-size:13px; margin-bottom:16px; }
  .envmeta { max-width:980px; }
  .legend-table { border-collapse:collapse; margin:0 0 18px; }
  .legend-table th, .legend-table td { padding:7px 14px; text-align:center; font-size:14px;
        border-bottom:1px solid var(--border); }
  .legend-table thead th { background:var(--card); }
  .legend-table .envname { text-align:left; font-weight:600; }
  .score { display:block; color:var(--muted); font-size:11px; font-weight:400; margin-top:2px; }
  .cell { background:var(--card); border:1px solid var(--border); border-radius:8px; padding:8px; display:inline-block; }
  .cell img { width:100%; max-width:1198px; border-radius:5px; display:block; }
  footer { color:var(--muted); font-size:12px; padding:24px; text-align:center; }
"""


def main():
    sel = json.load(open(SEL))
    p = []
    p.append('<!DOCTYPE html>')
    p.append('<html lang="en"><head><meta charset="utf-8">')
    p.append('<meta name="viewport" content="width=device-width, initial-scale=1">')
    p.append('<title>FedGuide — Policy Rollout Gallery</title>')
    p.append('<style>' + CSS + '</style></head><body>')
    p.append('<header><h1>FedGuide — Policy Rollout Gallery</h1>')
    p.append('<div class="sub">Round ' + str(sel.get("round", 0)) +
             ' · seed selection: ' + esc(sel.get("seed_selection", "")) + '</div></header>')

    envs = sel["environments"]

    # Legend table: selected seed + tail-10 eval return per environment × method.
    p.append('<section>')
    p.append('<div class="envmeta">Columns in the GIF are methods ('
             + ", ".join(label for _, label in METHODS)
             + '); rows are clients (c0…), grouped by environment. '
             'Selected seed and tail-10 eval return per cell:</div>')
    p.append('<table class="legend-table"><thead><tr><th>Environment</th>')
    for _, label in METHODS:
        p.append('<th>' + label + '</th>')
    p.append('</tr></thead><tbody>')
    for env, ed in envs.items():
        first = next(iter(ed["clients"].values()))
        p.append('<tr><td class="envname">' + esc(ed["display"]) +
                 '<span class="score">' + str(len(ed["clients"])) + ' clients</span></td>')
        for key, _ in METHODS:
            m = first.get(key, {})
            sc = m.get("score")
            scstr = ("%.2f" % sc) if isinstance(sc, (int, float)) else "—"
            p.append('<td>seed ' + str(m.get("seed")) +
                     '<span class="score">return ' + scstr + '</span></td>')
        p.append('</tr>')
    p.append('</tbody></table>')
    p.append('<div class="cell"><img src="all_envs.gif" '
             'alt="All environments × clients × methods"></div>')
    p.append('</section>')

    p.append('<footer>Generated from video_selection.json · '
             'single combined GIF: assets/gifs/all_envs.gif</footer>')
    p.append('</body></html>')

    with open(OUT, "w") as f:
        f.write("\n".join(p))
    print("wrote", OUT, os.path.getsize(OUT), "bytes")


if __name__ == "__main__":
    main()

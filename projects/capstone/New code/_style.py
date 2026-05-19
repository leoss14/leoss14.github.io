"""
_style.py — Shared Plotly style for all viz notebooks.

Every chart's title is `None` (the page supplies an h4). Colours match
the portfolio palette (navy / red on a #fafafa background).
"""

import os
import plotly.io as pio

HERE      = os.path.dirname(os.path.abspath(__file__))
OUT_DIR   = os.path.abspath(os.path.join(HERE, "..", "outputs"))
ARTIFACTS = os.path.abspath(os.path.join(HERE, "..", "artifacts"))
os.makedirs(OUT_DIR, exist_ok=True)

FONT  = "Public Sans, system-ui, -apple-system, sans-serif"
NAVY  = "#1a2744"
SUBTT = "#6b7280"
BG    = "#fafafa"
GRID  = "#e5e7eb"
CFG   = dict(displayModeBar=False, displaylogo=False, responsive=True)


def base_layout(**kw):
    d = dict(
        template="plotly_white",
        plot_bgcolor=BG,
        paper_bgcolor=BG,
        font=dict(family=FONT, size=12, color=NAVY),
        margin=dict(l=60, r=40, t=30, b=50),
        height=560,
    )
    d.update(kw)
    return d


def save(fig, name):
    """Write a Plotly figure to <project>/outputs/<name>.html."""
    path = os.path.join(OUT_DIR, name)
    fig.write_html(path, config=CFG, include_plotlyjs="cdn")
    print(f"  wrote outputs/{name}")


def hex_rgba(h, a):
    """Convert a hex colour like #1a2744 or short #abc to an rgba(...) string with alpha a.

    Resilient to:
      - 3-digit short hex (#abc expanded to #aabbcc)
      - leading/trailing whitespace
      - missing #
      - anything malformed -> neutral grey
    """
    s = (h or "").strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6:
        s = "999999"
    try:
        r, g, b = int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    except ValueError:
        r, g, b = 153, 153, 153
    return f"rgba({r},{g},{b},{a})"


def artifact_path(name):
    return os.path.join(ARTIFACTS, name)

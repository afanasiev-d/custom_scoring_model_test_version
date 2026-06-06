"""Shared visual identity for the Credit Scoring app.

A single, solid fintech look-and-feel (clean white canvas, deep-navy ink, teal
accent, green/red for good/bad) applied to every matplotlib/seaborn plot in the
app. Import the colour constants where needed and call :func:`setup` once at
start-up to apply the global theme.
"""
import io
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# ── Fintech palette ───────────────────────────────────────────────────────────
INK    = "#0A2540"   # primary text / deep navy (Stripe-style)
NAVY   = "#0A2540"
TEAL   = "#06B6D4"   # accent / model line
GOOD   = "#16A34A"   # "good" outcome (green)
BAD    = "#E11D48"   # "bad" outcome (rose-red)
GOLD   = "#F59E0B"   # highlights / cut-off markers
SLATE  = "#64748B"   # muted labels / reference lines
GRID   = "#E2E8F0"   # gridlines
BG     = "#FFFFFF"   # canvas
PANEL  = "#F8FAFC"   # subtle panel fill

PALETTE = [NAVY, TEAL, GOOD, GOLD, BAD, SLATE]

# Line / marker weights — kept deliberately light for an elegant, modern look.
LW     = 1.7   # main series
LW_REF = 1.0   # reference / dashed lines
MS     = 30    # scatter marker size

# Diverging colormap for correlation heatmaps: rose ↔ white ↔ navy.
CORR_CMAP = LinearSegmentedColormap.from_list(
    "fintech_corr", [BAD, "#FBE9EC", BG, "#E2ECF7", NAVY], N=256
)
def style_table(df, precision=4):
    """Return a pandas ``Styler`` for a clean, report-style table (no colours):
    centered navy column headers, centered numeric values, left-aligned text,
    tidy number formatting and a hidden index. Render with ``st.table``."""
    def _fmt(v):
        if isinstance(v, float):
            if pd.isna(v):
                return ""
            if float(v).is_integer():
                return f"{int(v):,}"
            return f"{v:,.{precision}f}".rstrip("0").rstrip(".")
        return v

    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    str_cols = [c for c in df.columns if c not in num_cols]

    sty = df.style.format(_fmt)
    sty = sty.set_table_styles([
        {'selector': 'th.col_heading',
         'props': [('color', NAVY), ('font-weight', '700'), ('text-align', 'center'),
                   ('background-color', '#F8FAFC'), ('border-bottom', f'2px solid {TEAL}'),
                   ('padding', '7px 12px')]},
        {'selector': 'td', 'props': [('padding', '5px 12px'), ('color', INK)]},
        {'selector': '', 'props': [('border-collapse', 'collapse')]},
    ])
    if num_cols:
        sty = sty.set_properties(subset=num_cols, **{'text-align': 'center'})
    if str_cols:
        sty = sty.set_properties(subset=str_cols, **{'text-align': 'left'})
    sty = sty.hide(axis='index')
    return sty


def _pick_font():
    """Prefer a clean modern sans-serif if available, else fall back gracefully."""
    available = {f.name for f in _fm.fontManager.ttflist}
    for name in ("Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"):
        if name in available:
            return name
    return "DejaVu Sans"


def setup():
    """Apply the global theme. Safe to call multiple times."""
    font = _pick_font()
    sns.set_theme(style="whitegrid", context="talk", font=font)
    plt.rcParams.update({
        "figure.facecolor": BG,
        "figure.edgecolor": BG,
        "savefig.facecolor": BG,
        "savefig.bbox": "tight",
        "axes.facecolor": BG,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.titlesize": 13.5,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.titlepad": 10,
        "axes.labelsize": 10,
        "axes.labelcolor": SLATE,
        "axes.labelweight": "medium",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": SLATE,
        "ytick.color": SLATE,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "text.color": INK,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "lines.linewidth": LW,
        "lines.markersize": 5,
        "lines.solid_capstyle": "round",
        "font.size": 9.5,
        "figure.dpi": 110,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
    })


# ── Figure gallery ────────────────────────────────────────────────────────────
# Plots register themselves here as high-resolution PNG bytes so the app can offer
# a single "download all visualizations" button at the end of the run.
_GALLERY = []


def reset_gallery():
    """Clear the collected figures (call once at the start of each model run)."""
    _GALLERY.clear()


def capture(name, fig, dpi=150):
    """Store a figure as a PNG in the gallery under ``name`` (order preserved)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor=BG)
    _GALLERY.append((name, buf.getvalue()))


def gallery_count():
    return len(_GALLERY)


def gallery_items():
    """Return the captured figures as a list of ``(name, png_bytes)`` in order."""
    return list(_GALLERY)


def gallery_zip():
    """Bundle all captured figures into a .zip, or return ``None`` if empty."""
    if not _GALLERY:
        return None
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in _GALLERY:
            zf.writestr(f"{name}.png", data)
    return out.getvalue()


def app_css():
    """Return the global CSS that gives the Streamlit app its fintech look."""
    return """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"],
.stMarkdown, .stMetric, button, input, textarea, label {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1340px; }
[data-testid="stHeader"] { background: transparent; }

/* Hero banner */
.hero {
    background: linear-gradient(120deg, #0A2540 0%, #0E5C73 62%, #06B6D4 135%);
    border-radius: 16px; padding: 1.5rem 1.8rem; margin-bottom: 1.4rem;
    box-shadow: 0 12px 30px rgba(10,37,64,0.22);
}
.hero h1 { color:#FFFFFF; font-size:1.8rem; font-weight:800; margin:0; letter-spacing:-0.02em; }
.hero p  { color:#CBD5E1; margin:.45rem 0 0; font-size:.98rem; }
.hero .pill { display:inline-block; background:rgba(255,255,255,0.14); color:#E2F6FA;
    border:1px solid rgba(255,255,255,0.28); padding:.16rem .7rem; border-radius:999px;
    font-size:.78rem; font-weight:600; margin-top:.7rem; }

/* Headings */
h2, h3 { color:#0A2540; font-weight:700; letter-spacing:-0.01em; }

/* Metric cards */
[data-testid="stMetric"] {
    background:#F8FAFC; border:1px solid #E2E8F0; border-left:5px solid #06B6D4;
    border-radius:12px; padding:14px 18px; box-shadow:0 1px 3px rgba(2,6,23,0.05);
}
[data-testid="stMetricValue"] { color:#0A2540; font-weight:800; }
[data-testid="stMetricLabel"] p { color:#64748B; font-weight:600; }

/* Buttons */
.stButton button, .stDownloadButton button, [data-testid="stFormSubmitButton"] button {
    border-radius:10px; font-weight:600; transition:all .15s ease;
}
[data-testid="stFormSubmitButton"] button {
    background:#0A2540; color:#FFFFFF; border:0; padding:.55rem 1.5rem; font-size:1rem;
    box-shadow:0 6px 16px rgba(10,37,64,0.25);
}
[data-testid="stFormSubmitButton"] button:hover { background:#06B6D4; color:#0A2540; }
.stDownloadButton button { border:1.5px solid #06B6D4; color:#0A2540; background:#FFFFFF; }
.stDownloadButton button:hover { background:#06B6D4; color:#FFFFFF; border-color:#06B6D4; }

/* Sidebar */
[data-testid="stSidebar"] { background:#F8FAFC; border-right:1px solid #E2E8F0; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color:#0A2540; }

/* Status / alerts / tables */
[data-testid="stStatus"] { border-radius:12px; border:1px solid #E2E8F0; }
[data-testid="stAlert"] { border-radius:10px; }
[data-testid="stDataFrame"] { border-radius:10px; overflow:hidden; }
</style>
"""


def title(ax, text, subtitle=None):
    """Left-aligned bold title with an optional muted subtitle, dashboard-style."""
    if subtitle:
        ax.set_title("")
        ax.text(0.0, 1.075, text, transform=ax.transAxes, color=INK,
                fontsize=13, fontweight="bold", va="bottom", ha="left")
        ax.text(0.0, 1.015, subtitle, transform=ax.transAxes, color=SLATE,
                fontsize=8.5, fontweight="normal", va="bottom", ha="left")
    else:
        ax.set_title(text, loc="left", color=INK, fontweight="bold", pad=10)

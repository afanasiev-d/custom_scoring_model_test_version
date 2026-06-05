"""Shared visual identity for the Credit Scoring app.

A single, solid fintech look-and-feel (clean white canvas, deep-navy ink, teal
accent, green/red for good/bad) applied to every matplotlib/seaborn plot in the
app. Import the colour constants where needed and call :func:`setup` once at
start-up to apply the global theme.
"""
import io
import zipfile
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

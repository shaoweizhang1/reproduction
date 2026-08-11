"""Shared matplotlib styling.

Colours follow Figure 3 of the paper -- red/green for the CNN,
orange/blue for DeepProbLog -- keyed by series name so a model keeps its
colour across figures.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "none"
INK = "#8e8d88"
INK_SOFT = "#8e8d88"
INK_MUTED = "#8e8d88"
GRID = "#8e8d8844"
AXIS = "#8e8d8899"

SERIES_COLORS = {
    "DeepProbLog": ("#e8912f", "#2a78d6"),
    "CNN (concat)": ("#d6392a", "#1baf7a"),
    "CNN (separate)": ("#a56ad0", "#0f8f9e"),
}
FALLBACK_COLORS = ("#8a8880", "#eda100")

def colors_for(label):
    return SERIES_COLORS.get(label, FALLBACK_COLORS)

def apply_style():
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "figure.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.titlesize": 12,
            "axes.titlecolor": INK,
            "axes.labelcolor": INK_SOFT,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 1,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 1,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "font.size": 10,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 1.6,
        }
    )

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "/home/claude/portfolio/figs/"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.30,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "legend.fontsize": 8.5,
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "0.75",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 190,
    "savefig.dpi": 190,
    "savefig.bbox": "tight",
    "axes.prop_cycle": plt.cycler(color=[
        "#1f4e79", "#c0392b", "#1e8449", "#8e44ad",
        "#d68910", "#2874a6", "#7b241c", "#117864"]),
})


def save(fig, name):
    path = FIG + name
    fig.savefig(path)
    plt.close(fig)
    print("  figure ->", name)

"""Shared style module for presentation figures."""
import matplotlib.pyplot as plt
from matplotlib.font_manager import findfont, FontProperties

# Colors
PRIMARY   = '#4A7FB5'  # steel blue — primary data
SECONDARY = '#A0937D'  # warm gray — secondary/comparison
DARK      = '#2D2D2D'  # charcoal — titles, labels
MID       = '#777777'  # medium gray — annotations
GRID      = '#E0E0E0'  # light gray — gridlines
BG        = '#FFFFFF'   # white — always

def setup():
    """Apply full rcParams preamble. Returns detected font name."""
    font = 'Calibri'
    try:
        if 'calibri' not in findfont(FontProperties(family='Calibri')).lower():
            raise ValueError
    except Exception:
        font = 'DejaVu Sans'

    rc = {
        'font.family': font, 'font.size': 14,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': MID, 'axes.labelcolor': DARK,
        'xtick.color': MID, 'ytick.color': MID, 'text.color': DARK,
        'figure.facecolor': BG, 'axes.facecolor': BG,
        'savefig.facecolor': BG, 'savefig.dpi': 300,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.15,
    }
    plt.rcParams.update(rc)
    return font


def annotate_bracket(ax, y, x_left, x_right, label, *, offset_y=12,
                     fontsize=11.5, color=MID):
    """Draw a bracket between x_left and x_right at height y with label above."""
    mid = (x_left + x_right) / 2
    tick = 0.12  # vertical tick height (axes fraction-ish, small)
    ax.plot([x_left, x_left], [y, y + tick], color=color, lw=1.0, zorder=3)
    ax.plot([x_right, x_right], [y, y + tick], color=color, lw=1.0, zorder=3)
    ax.plot([x_left, x_right], [y + tick, y + tick], color=color, lw=1.0,
            zorder=3)
    ax.annotate(label, xy=(mid, y + tick), xycoords='data',
                xytext=(0, offset_y), textcoords='offset points',
                ha='center', va='bottom', fontsize=fontsize, color=color,
                linespacing=1.3)


def bar_label(ax, x, y, value, *, fmt='{:.1f}%', offset=(6, 0),
              fontsize=13, color=DARK, **kwargs):
    """Place a value label at a bar tip using offset points."""
    ax.annotate(fmt.format(value), xy=(x, y), xycoords='data',
                xytext=offset, textcoords='offset points',
                ha='left', va='center', fontsize=fontsize, color=color,
                fontweight='bold', **kwargs)

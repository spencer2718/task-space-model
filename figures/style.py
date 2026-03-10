"""Shared style module for presentation figures."""
import matplotlib.pyplot as plt
from matplotlib.font_manager import findfont, FontProperties
from matplotlib.ticker import FuncFormatter

# ─── Core semantic colors ───
PRIMARY   = '#4A7FB5'  # steel blue — "ours" / embedding data
SECONDARY = '#A0937D'  # warm tan — "theirs" / O*NET / comparison

# ─── Extended palette (multi-category plots) ───
RED       = '#C75C5C'  # muted red
ORANGE    = '#D4845A'  # amber
GREEN     = '#44AA99'  # teal (colorblind-safe, Tol palette)
PURPLE    = '#8B6BAE'  # muted purple

# ─── Neutrals ───
DARK      = '#2D2D2D'  # charcoal — titles, occupation labels
MID       = '#777777'  # medium gray — annotations, subtitles
GRID      = '#E0E0E0'  # light gray — gridlines, separators
BG        = '#FFFFFF'

# ─── Font size scale ───
FONT_TITLE = 11        # panel titles, occupation headers
FONT_LABEL = 9         # axis labels, bar labels
FONT_TICK  = 8         # tick labels, annotation text
FONT_NOTE  = 7.5       # subtitles, column headers, footnotes


def lighten(hex_color, factor=0.85):
    """Mix hex_color toward white. factor=0 returns original, factor=1 returns white."""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f'#{r:02x}{g:02x}{b:02x}'


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


def add_subtitle(fig, text, y=-0.03, fontsize=None):
    """Add italic caption below the plot area."""
    fig.text(0.5, y, text, ha='center',
             fontsize=fontsize or FONT_TICK,
             color=MID, fontstyle='italic')


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
              fontsize=None, color=DARK, **kwargs):
    """Place a value label at a bar tip using offset points."""
    ax.annotate(fmt.format(value), xy=(x, y), xycoords='data',
                xytext=offset, textcoords='offset points',
                ha='left', va='center',
                fontsize=fontsize or FONT_TITLE,
                color=color, fontweight='bold', **kwargs)


def format_log_ticks(ax, axis='y'):
    """Replace scientific notation with plain integers on a log-scale axis."""
    formatter = FuncFormatter(lambda x, _: f'{int(x):,}' if x >= 1 else '')
    if axis == 'y':
        ax.yaxis.set_major_formatter(formatter)
    else:
        ax.xaxis.set_major_formatter(formatter)

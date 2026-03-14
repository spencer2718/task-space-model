"""
Figure 7 — CPS Data Pipeline (funnel diagram)
Target: Slide B1 (backup — Data Pipeline)
Shows sample construction from raw CPS to final transitions.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from figures.style import setup, PRIMARY, DARK, MID, GRID, FONT_TITLE, FONT_TICK

font = setup()

# === Pipeline stages (from paper Table 1) ===
stages = [
    ("Raw CPS\nextract",        10_072_119, "100%"),
    ("Employment\nfilter",       4_407_432, "43.8%"),
    ("Consecutive\nmonths",      2_353_725, "23.4%"),
    ("Raw\ntransitions",           157_606, "1.6%"),
    ("Persistence\nfilter",        106_116, "1.1%"),
    ("O*NET\nmapping",              89_329, "0.9%"),
]

max_count = stages[0][1]

# === Figure ===
fig, ax = plt.subplots(figsize=(9.0, 4.2))
ax.set_xlim(-0.5, len(stages) - 0.5)
ax.set_ylim(-1.5, 2.5)
ax.axis('off')

# Bar parameters
bar_max_height = 3.0
bar_width = 0.7
y_center = 0.5

for i, (label, count, pct) in enumerate(stages):
    # Height proportional to log scale (otherwise later bars invisible)
    # Use sqrt for visual scaling
    frac = count / max_count
    height = max(bar_max_height * (frac ** 0.35), 0.3)

    x = i
    bottom = y_center - height / 2

    # Color gradient: darken as pipeline progresses
    alpha = 0.4 + 0.5 * (i / (len(stages) - 1))

    bar = mpatches.FancyBboxPatch(
        (x - bar_width / 2, bottom), bar_width, height,
        boxstyle="round,pad=0.05", facecolor=PRIMARY, alpha=alpha,
        edgecolor='none', zorder=2,
    )
    ax.add_patch(bar)

    # Count label above bar
    ax.text(x, y_center + height / 2 + 0.15,
            f'{count:,.0f}', ha='center', va='bottom',
            fontsize=FONT_TITLE, fontweight='bold', color=DARK)

    # Stage label below bar
    ax.text(x, y_center - height / 2 - 0.15,
            label, ha='center', va='top',
            fontsize=FONT_TICK, color=MID, linespacing=1.1)

    # Arrow between stages
    if i < len(stages) - 1:
        ax.annotate('', xy=(x + 0.55, y_center), xytext=(x + 0.45, y_center),
                     arrowprops=dict(arrowstyle='simple,head_width=3,head_length=2', fc=GRID, ec='none'))

# Final count highlight
final_x = len(stages) - 1
ax.text(final_x, y_center - 1.3,
        '89,329 verified transitions',
        ha='center', va='center', fontsize=FONT_TITLE, fontweight='bold',
        color=PRIMARY)

plt.tight_layout()
plt.savefig('figures/fig7_sankey_pipeline.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"Saved figures/fig7_sankey_pipeline.png  (font: {font})")

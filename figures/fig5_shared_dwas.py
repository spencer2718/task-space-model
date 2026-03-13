"""
Figure 5 — DWA overlap diagram: Budget Analysts vs Credit Analysts
Target: Slide 4 (left panel)
Shows shared and unshared DWAs between two finance occupations.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from figures.style import setup, PRIMARY, SECONDARY, DARK, MID, GRID, FONT_TITLE, FONT_NOTE

font = setup()

# === Data: Budget Analysts vs Credit Analysts ===
shared_dwas = [
    "Advise others on\nfinancial matters",
    "Analyze business\nor financial data",
    "Prepare financial\ndocuments or budgets",
]

left_dwas = [
    "Analyze budgetary\nor accounting data",
    "Discuss strategies\nwith managers",
    "Verify accuracy of\nfinancial information",
]

right_dwas = [
    "Assess financial\nstatus of clients",
    "Analyze market\nconditions or trends",
    "Examine\nfinancial records",
]

# === Layout ===
fig, ax = plt.subplots(figsize=(4.3, 4.2))
ax.set_xlim(0, 10)
ax.set_ylim(3.5, 10)
ax.axis('off')

# Node drawing helper
def draw_node(ax, x, y, text, color, width=2.6, height=0.7):
    box = mpatches.FancyBboxPatch(
        (x - width / 2, y - height / 2), width, height,
        boxstyle="round,pad=0.15", facecolor=color, edgecolor='none',
        alpha=0.15, zorder=1,
    )
    ax.add_patch(box)
    border = mpatches.FancyBboxPatch(
        (x - width / 2, y - height / 2), width, height,
        boxstyle="round,pad=0.15", facecolor='none', edgecolor=color,
        linewidth=0.8, alpha=0.5, zorder=2,
    )
    ax.add_patch(border)
    ax.text(x, y, text, ha='center', va='center', fontsize=6.5,
            color=DARK, zorder=3, linespacing=1.1)

# Occupation labels
ax.text(1.8, 9.3, 'Budget Analysts', ha='center', va='center',
        fontsize=FONT_TITLE, fontweight='bold', color=PRIMARY)
ax.text(8.2, 9.3, 'Credit Analysts', ha='center', va='center',
        fontsize=FONT_TITLE, fontweight='bold', color=SECONDARY)

# Node y-positions (3 rows)
ys = [7.4, 6.2, 5.0]

# Left-only DWAs (Budget Analysts)
for dwa, y in zip(left_dwas, ys):
    draw_node(ax, 1.8, y, dwa, PRIMARY)

# Right-only DWAs (Credit Analysts)
for dwa, y in zip(right_dwas, ys):
    draw_node(ax, 8.2, y, dwa, SECONDARY)

# Shared DWAs (center column)
for dwa, y in zip(shared_dwas, ys):
    draw_node(ax, 5.0, y, dwa, MID, width=2.8)

# Vertical column separators
sep_top = max(ys) + 0.5
sep_bot = min(ys) - 0.5
ax.plot([3.35, 3.35], [sep_bot, sep_top], color=GRID, lw=0.8, ls='--', zorder=0)
ax.plot([6.65, 6.65], [sep_bot, sep_top], color=GRID, lw=0.8, ls='--', zorder=0)

# Column headers
ax.text(1.8, 8.35, '7 unique', ha='center', va='center',
        fontsize=FONT_NOTE, color=PRIMARY, fontstyle='italic')
ax.text(5.0, 8.35, '3 shared', ha='center', va='center',
        fontsize=FONT_NOTE, color=MID, fontstyle='italic')
ax.text(8.2, 8.35, '9 unique', ha='center', va='center',
        fontsize=FONT_NOTE, color=SECONDARY, fontstyle='italic')

# Vertical ellipses indicating more DWAs exist
ax.text(1.8, 8.05, '⋮', ha='center', va='center', fontsize=10, color=PRIMARY)
ax.text(1.8, 4.25, '⋮', ha='center', va='center', fontsize=10, color=PRIMARY)
ax.text(8.2, 8.05, '⋮', ha='center', va='center', fontsize=10, color=SECONDARY)
ax.text(8.2, 4.25, '⋮', ha='center', va='center', fontsize=10, color=SECONDARY)

# Subtitle
ax.text(5.0, 3.8,
        'Only 3 of 19 DWAs overlap — O*NET sees these as distant (d = 0.70)',
        ha='center', va='center', fontsize=FONT_NOTE, color=MID, fontstyle='italic')

plt.tight_layout()
plt.savefig('figures/fig5_shared_dwas.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"Saved figures/fig5_shared_dwas.png  (font: {font})")

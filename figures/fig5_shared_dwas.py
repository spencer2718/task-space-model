"""
Figure 5 — DWA overlap diagram: Pipelayers vs Cement Masons
Target: Slide 4 (left panel)
Shows shared and unshared DWAs between two construction occupations.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from figures.style import setup, PRIMARY, SECONDARY, DARK, MID, GRID

font = setup()

# === Data (hardcoded from O*NET Tasks-to-DWAs mapping) ===
shared_dwas = [
    "Compact materials to\ncreate level bases",
    "Cut metal components\nfor installation",
    "Direct construction or\nextraction personnel",
    "Drill holes in\nconstruction materials",
]

pipe_dwas = [
    "Install plumbing\nor piping",
    "Dig holes\nor trenches",
    "Weld metal\ncomponents",
    "Drive trucks or\ntruck-mounted equip.",
]

cement_dwas = [
    "Finish concrete\nsurfaces",
    "Install masonry\nmaterials",
    "Apply sealants or\nprotective coatings",
    "Position construction\nforms or molds",
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
ax.text(1.8, 9.3, 'Pipelayers', ha='center', va='center',
        fontsize=12, fontweight='bold', color=PRIMARY)
ax.text(8.2, 9.3, 'Cement Masons', ha='center', va='center',
        fontsize=12, fontweight='bold', color=SECONDARY)

# Pipelayer-only DWAs (left column)
pipe_ys = [7.8, 6.8, 5.8, 4.8]
for dwa, y in zip(pipe_dwas, pipe_ys):
    draw_node(ax, 1.8, y, dwa, PRIMARY)

# Cement Mason-only DWAs (right column)
cement_ys = [7.8, 6.8, 5.8, 4.8]
for dwa, y in zip(cement_dwas, cement_ys):
    draw_node(ax, 8.2, y, dwa, SECONDARY)

# Shared DWAs (center column)
shared_ys = [7.8, 6.8, 5.8, 4.8]
for dwa, y in zip(shared_dwas, shared_ys):
    draw_node(ax, 5.0, y, dwa, MID, width=2.8)
    # Lines to Pipelayers
    ax.plot([3.1, 3.6], [y, y], color=GRID, lw=0.8, zorder=0)
    # Lines to Cement Masons
    ax.plot([6.4, 6.9], [y, y], color=GRID, lw=0.8, zorder=0)

# Column headers
ax.text(1.8, 8.6, 'unique', ha='center', va='center',
        fontsize=7.5, color=PRIMARY, fontstyle='italic')
ax.text(5.0, 8.6, 'shared', ha='center', va='center',
        fontsize=7.5, color=MID, fontstyle='italic')
ax.text(8.2, 8.6, 'unique', ha='center', va='center',
        fontsize=7.5, color=SECONDARY, fontstyle='italic')

# Vertical ellipses indicating more DWAs exist
ax.text(1.8, 8.25, '⋮', ha='center', va='center', fontsize=10, color=PRIMARY)
ax.text(1.8, 4.35, '⋮', ha='center', va='center', fontsize=10, color=PRIMARY)
ax.text(8.2, 8.25, '⋮', ha='center', va='center', fontsize=10, color=SECONDARY)
ax.text(8.2, 4.35, '⋮', ha='center', va='center', fontsize=10, color=SECONDARY)

# Subtitle
ax.text(5.0, 3.8,
        'Only 4 of 40 DWAs overlap — O*NET sees these as distant (d = 0.88)',
        ha='center', va='center', fontsize=7.5, color=MID, fontstyle='italic')

plt.tight_layout()
plt.savefig('figures/fig5_shared_dwas.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"Saved figures/fig5_shared_dwas.png  (font: {font})")

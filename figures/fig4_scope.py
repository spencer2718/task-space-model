"""
Figure 4 — Supply-demand decomposition (scope)
Target: Slide 7 (What the Measure Captures)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
from figures.style import setup, PRIMARY, SECONDARY, DARK, MID, add_subtitle

font = setup()

labels = ["Demand\n(BLS openings)", "Per-origin\ngeometry", "Aggregate\ngeometry"]
values = [0.798, 0.128, 0.043]
colors = [SECONDARY, PRIMARY, PRIMARY]

fig, ax = plt.subplots(figsize=(4.0, 3.0))

bars = ax.bar(range(len(labels)), values, color=colors, width=0.6,
              edgecolor='none', zorder=2)

# Value labels above bars
for i, (v, bar) in enumerate(zip(values, bars)):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
            f'{v:.2f}', ha='center', va='bottom',
            fontsize=11, color=DARK, fontweight='bold')

ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=9, color=DARK)
ax.set_ylabel('Spearman ρ', fontsize=11, color=MID)
ax.set_ylim(0, 1.0)
ax.tick_params(axis='y', labelsize=9)
ax.tick_params(axis='x', length=0)

plt.tight_layout()
add_subtitle(fig, 'Geometry ranks destinations correctly (MPR = 0.74)\nbut does not predict aggregate flows', y=-0.02)
plt.savefig('figures/fig4_scope.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"Saved figures/fig4_scope.png  (font: {font})")

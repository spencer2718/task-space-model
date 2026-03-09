"""
Figure 4 — Supply-Demand Decomposition (scope)
Target: Slide 7 ("What the Measure Captures")
Data: Hardcoded from outputs/experiments/demand_probe_decomposition_v0703b.json
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
from figures.style import setup, bar_label
from figures.style import PRIMARY, SECONDARY, DARK, MID

font = setup()

# === Data ===
metrics = [
    ("Demand only\n(BLS openings → aggregate inflows)",           0.798),
    ("Per-origin geometry\n(embedding distance → destination ranking)", 0.128),
    ("Aggregate geometry\n(embedding distance → total inflows)",   0.043),
]

# Reverse so top bar is demand (barh draws bottom-up)
labels = [m[0] for m in metrics][::-1]
values = [m[1] for m in metrics][::-1]
colors = [PRIMARY, PRIMARY, SECONDARY]  # bottom two are geometry (PRIMARY), top is demand

# === Figure ===
bar_height = 0.55
positions = [0, 1.2, 2.4]

fig, ax = plt.subplots(figsize=(4.0, 3.8))

bars = ax.barh(positions, values, color=colors, height=bar_height,
               edgecolor='none', zorder=2)

# Spec name labels
for pos, label in zip(positions, labels):
    ax.text(-0.02, pos, label, ha='right', va='center', fontsize=9,
            color=DARK, fontweight='medium')

# Value labels
for pos, val in zip(positions, values):
    bar_label(ax, val, pos, val, fmt='{:.2f}', fontsize=11)

# Axes
ax.set_xlim(0, 1.15)
ax.set_ylim(-0.6, 3.2)
ax.set_xticks([0, 0.5, 1.0])
ax.set_xticklabels(['0', '0.50', '1.00'], fontsize=9)
ax.set_yticks([])
ax.set_xlabel("Spearman ρ", fontsize=11, labelpad=8)
ax.tick_params(axis='y', length=0)
ax.tick_params(axis='x', length=4)

plt.tight_layout()
plt.savefig('figures/fig4_scope.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"Saved figures/fig4_scope.png  (font: {font})")

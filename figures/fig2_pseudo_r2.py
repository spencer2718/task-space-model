"""
Figure 2 — Pseudo-R² comparison bar chart
Target: Slide 5 ("Main Result: Embeddings Dominate")
Data: Hardcoded from paper Tables 2 & 3
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
from figures.style import setup, bar_label
from figures.style import PRIMARY, SECONDARY, DARK, MID

font = setup()

# === Data (corrected v0.7.7.0+) ===
specs = [
    ("Embedding + Centroid",    14.08),
    ("Embedding + Wasserstein", 13.76),
    ("O*NET + Cosine",           8.05),
    ("O*NET + Euclidean",        6.06),
]

# Reverse so best is at top (barh draws bottom-up)
labels = [s[0] for s in specs][::-1]
values = [s[1] for s in specs][::-1]
colors = [SECONDARY, SECONDARY, PRIMARY, PRIMARY]

# Bar positions — even spacing
bar_height = 0.55
positions = [0, 1.2, 2.4, 3.6]

# === Figure ===
fig, ax = plt.subplots(figsize=(9.0, 5.0))

bars = ax.barh(positions, values, color=colors, height=bar_height,
               edgecolor='none', zorder=2)

# Spec name labels
for pos, label in zip(positions, labels):
    ax.text(-0.3, pos, label, ha='right', va='center', fontsize=14,
            color=DARK, fontweight='medium')

# Value labels using style helper
for pos, val in zip(positions, values):
    bar_label(ax, val, pos, val)

# === Vertical bracket: O*NET Cosine → Embedding Centroid ===
y_bottom = positions[1]  # O*NET Cosine
y_top = positions[3]     # Embedding Centroid
x_bracket = 16.5

ax.plot([x_bracket, x_bracket], [y_bottom, y_top], color=MID, lw=1.0, zorder=3)
ax.plot([x_bracket - 0.3, x_bracket], [y_bottom, y_bottom], color=MID, lw=1.0, zorder=3)
ax.plot([x_bracket - 0.3, x_bracket], [y_top, y_top], color=MID, lw=1.0, zorder=3)
ax.annotate('Embedding\nrepresentation\n→ ~75%',
            xy=(x_bracket, (y_bottom + y_top) / 2), xycoords='data',
            xytext=(8, 0), textcoords='offset points',
            ha='left', va='center', fontsize=11.5, color=MID,
            linespacing=1.3)

# === Axes ===
ax.set_xlim(0, 22)
ax.set_ylim(-0.6, 4.8)
ax.set_xticks([0, 5, 10, 15])
ax.set_xticklabels(['0%', '5%', '10%', '15%'], fontsize=12)
ax.set_yticks([])
ax.set_xlabel("McFadden's pseudo-R²", fontsize=13, labelpad=8)
ax.tick_params(axis='y', length=0)
ax.tick_params(axis='x', length=4)

plt.tight_layout()
plt.savefig('figures/fig2_pseudo_r2.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"Saved figures/fig2_pseudo_r2.png  (font: {font})")

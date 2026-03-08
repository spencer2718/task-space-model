"""
Figure 2 — Pseudo-R² comparison bar chart
Target: Slide 5 ("Main Result: Embeddings Dominate")
Data: Hardcoded from paper Tables 2 & 3
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
from figures.style import setup, annotate_bracket, bar_label
from figures.style import PRIMARY, SECONDARY, DARK, MID

font = setup()

# === Data ===
specs = [
    ("Embedding + Wasserstein", 14.51),
    ("Embedding + Centroid",    14.08),
    ("O*NET + Cosine",           8.05),
    ("O*NET + Euclidean",        6.06),
]
ground_identity  = 7.42
ground_embedding = 14.51

# Reverse so best is at top (barh draws bottom-up)
labels = [s[0] for s in specs][::-1]
values = [s[1] for s in specs][::-1]
colors = [SECONDARY, SECONDARY, PRIMARY, PRIMARY]

# Bar positions — extra spacing between top two bars for ghost bar annotation
bar_height = 0.55
positions = [0, 1.2, 2.4, 3.8]  # wider gap before top bar

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

# === Ghost bar for identity ground metric ===
top_y = positions[-1]
ax.barh(top_y, ground_identity, height=bar_height, color='none',
        edgecolor=MID, linewidth=1.2, linestyle='--', zorder=1)

# Identity label — below the ghost bar with white background pad
label_y = top_y - (bar_height / 2) - 0.15
ax.annotate(f'{ground_identity:.1f}%  (identity ground metric)',
            xy=(ground_identity, label_y), xycoords='data',
            xytext=(6, 0), textcoords='offset points',
            ha='left', va='center', fontsize=11.5, color=MID, style='italic',
            bbox=dict(facecolor='white', edgecolor='none', pad=2),
            zorder=4)

# Bracket annotation using style helper
pct_improvement = (ground_embedding - ground_identity) / ground_identity * 100
annotate_bracket(ax, y=top_y + 0.38, x_left=ground_identity + 0.15,
                 x_right=ground_embedding + 0.15,
                 label=f'Same Wasserstein, different\nground metric  →  +{pct_improvement:.0f}%',
                 offset_y=10, fontsize=11.5)

# === Axes ===
ax.set_xlim(0, 18.5)
ax.set_ylim(-0.6, 5.2)
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

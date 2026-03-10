"""
Figure 6 — Embedding vs O*NET distance histograms
Target: Slide 4 (right panel)
Shows why embeddings disagree with O*NET for Budget Analysts vs Credit Analysts.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt

from figures.style import (setup, PRIMARY, SECONDARY, DARK, MID, GRID, format_log_ticks,
                           add_subtitle, FONT_TITLE, FONT_LABEL, FONT_TICK)

font = setup()

# === Load distance matrices ===
D_embed = np.load('.cache/artifacts/v1/mobility/d_cosine_embed_census.npz')['distances']
D_onet = np.load('.cache/artifacts/v1/mobility/d_cosine_onet_census.npz')['distances']
codes = list(np.load('.cache/artifacts/v1/mobility/d_cosine_embed_census.npz')['census_codes'].astype(int))

budget_idx = codes.index(820)   # Budget Analysts
credit_idx = codes.index(830)   # Credit Analysts

d_embed_pair = D_embed[budget_idx, credit_idx]
d_onet_pair = D_onet[budget_idx, credit_idx]

print(f"Budget-Credit Analysts: embed={d_embed_pair:.4f}, onet={d_onet_pair:.4f}")

# Upper triangle values
n = D_embed.shape[0]
triu_idx = np.triu_indices(n, k=1)
embed_vals = D_embed[triu_idx]
onet_vals = D_onet[triu_idx]

print(f"Pairs: {len(embed_vals)}")
print(f"O*NET: {(onet_vals > 0.99).mean():.1%} at max distance")

# === Figure: two stacked panels ===
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.5, 4.0), sharex=False)
fig.subplots_adjust(hspace=0.45)

# --- Top panel: O*NET Cosine ---
ax1.hist(onet_vals, bins=60, color=SECONDARY, alpha=0.7, edgecolor='none')
ax1.set_yscale('log')
ax1.set_ylim(bottom=0.8)
ax1.axvline(d_onet_pair, color=DARK, linestyle='--', linewidth=1.2, zorder=3)
fig.canvas.draw()
ax1.annotate(f'Budget\u2013Credit: {d_onet_pair:.2f}',
             xy=(d_onet_pair, 100),
             xytext=(-15, 30), textcoords='offset points',
             fontsize=FONT_TICK, color=DARK, fontweight='bold', ha='right',
             arrowprops=dict(arrowstyle='->', color=DARK, lw=0.8))
ax1.set_title("O*NET Cosine Distance", fontsize=FONT_TITLE, color=DARK, pad=6)
ax1.set_ylabel('Pairs', fontsize=FONT_TICK)
ax1.set_xlim(0, 1.05)
format_log_ticks(ax1)
ax1.set_yticks([1, 10, 100, 1000, 10000, 100000])
ax1.tick_params(axis='both', labelsize=FONT_TICK)

# --- Bottom panel: Embedding Centroid ---
ax2.hist(embed_vals, bins=60, color=PRIMARY, alpha=0.7, edgecolor='none')
ax2.axvline(d_embed_pair, color=DARK, linestyle='--', linewidth=1.2, zorder=3)
# Need to draw first to get ylim
fig.canvas.draw()
ax2.annotate(f'Budget\u2013Credit: {d_embed_pair:.2f}',
             xy=(d_embed_pair, ax2.get_ylim()[1] * 0.5),
             xytext=(40, 20), textcoords='offset points',
             fontsize=FONT_TICK, color=DARK, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=DARK, lw=0.8))
ax2.set_title("Embedding Centroid Distance", fontsize=FONT_TITLE, color=DARK, pad=6)
ax2.set_ylabel('Pairs', fontsize=FONT_TICK)
ax2.set_xlabel('Distance', fontsize=FONT_LABEL)
ax2.set_xlim(0, 1.05)
ax2.tick_params(axis='both', labelsize=FONT_TICK)

add_subtitle(fig, '78% of O*NET pairs at max distance — embeddings recover a usable distribution', y=-0.04)

plt.savefig('figures/fig6_embedding_similarity.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"Saved figures/fig6_embedding_similarity.png  (font: {font})")

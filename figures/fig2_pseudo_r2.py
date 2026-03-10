"""
Figure 2 — Pseudo-R² comparison bar chart
Target: Slide 5 ("Main Result: Embeddings Dominate")
Data: Hardcoded from paper Tables 2 & 3
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
from figures.style import (setup, bar_label, PRIMARY, SECONDARY, DARK, MID,
                           FONT_TITLE, FONT_LABEL, FONT_TICK)

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
positions = [0, 1.0, 2.0, 3.0]

# === Figure ===
fig, ax = plt.subplots(figsize=(9.0, 4.2))

bars = ax.barh(positions, values, color=colors, height=bar_height,
               edgecolor='none', zorder=2)

# Spec name labels
for pos, label in zip(positions, labels):
    ax.text(-0.3, pos, label, ha='right', va='center', fontsize=FONT_TITLE,
            color=DARK, fontweight='medium')

# Value labels using style helper
for pos, val in zip(positions, values):
    bar_label(ax, val, pos, val)

# === Axes (set before measuring text) ===
ax.set_xlim(0, 22)
ax.set_ylim(-0.6, 3.8)
ax.set_xticks([0, 5, 10, 15])
ax.set_xticklabels(['0%', '5%', '10%', '15%'], fontsize=FONT_TICK)
ax.set_yticks([])
ax.set_xlabel("McFadden's pseudo-R²", fontsize=FONT_TITLE, labelpad=8)
ax.tick_params(axis='y', length=0)
ax.tick_params(axis='x', length=4)

plt.tight_layout()

# === Vertical bracket: O*NET Cosine → Embedding Centroid ===
# Render first pass to measure value label text extents
fig.canvas.draw()
renderer = fig.canvas.get_renderer()

# Find the value label annotations (the bold percentage texts)
# They were created by bar_label() which calls ax.annotate()
# Collect all annotations, match by position to identify the two we need
bracket_bars = {
    'top': positions[3],     # Embedding Centroid
    'bottom': positions[1],  # O*NET Cosine
}

label_right_edges = {}
import matplotlib.text as mtext
for child in ax.get_children():
    if not isinstance(child, mtext.Annotation):
        continue
    try:
        text_str = child.get_text()
        if '%' not in text_str:
            continue
        # Annotation's xy attribute holds the data-coordinate anchor
        y_data = child.xy[1]
        # Match to bracket bars by y position
        for key, bar_y in bracket_bars.items():
            if abs(y_data - bar_y) < 0.3:
                # Get right edge of text in data coords
                bbox = child.get_window_extent(renderer)
                data_bbox = ax.transData.inverted().transform(bbox)
                label_right_edges[key] = data_bbox[1, 0]  # right x in data coords
    except Exception:
        continue

gap = 0.4       # small gap between label text and start of horizontal tick
bracket_ext = 1.5  # how far the vertical line extends beyond the top label

if 'top' in label_right_edges and 'bottom' in label_right_edges:
    y_bottom = bracket_bars['bottom']
    y_top = bracket_bars['top']

    # Vertical line sits well past the rightmost (top) label
    x_bracket = label_right_edges['top'] + gap + bracket_ext

    # Each horizontal tick starts just after its own label
    x_tick_top_start = label_right_edges['top'] + gap
    x_tick_bottom_start = label_right_edges['bottom'] + gap

    # Vertical line
    ax.plot([x_bracket, x_bracket], [y_bottom, y_top], color=MID, lw=1.0, zorder=3)
    # Top tick
    ax.plot([x_tick_top_start, x_bracket], [y_top, y_top], color=MID, lw=1.0, zorder=3)
    # Bottom tick (longer, spans the whitespace)
    ax.plot([x_tick_bottom_start, x_bracket], [y_bottom, y_bottom], color=MID, lw=1.0, zorder=3)

    ax.annotate('Embedding\nrepresentation\n+74.9%',
                xy=(x_bracket, (y_bottom + y_top) / 2), xycoords='data',
                xytext=(8, 0), textcoords='offset points',
                ha='left', va='center', fontsize=FONT_LABEL, color=MID,
                linespacing=1.3)

    ax.set_xlim(0, x_bracket + 6.5)
else:
    print("WARNING: Could not find label text extents, bracket not drawn")

plt.savefig('figures/fig2_pseudo_r2.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"Saved figures/fig2_pseudo_r2.png  (font: {font})")

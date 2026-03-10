"""
Figure 8 — Embedding flowchart
Target: Slide 3 (left panel) — What is a Sentence Embedding?
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from figures.style import (setup, PRIMARY, DARK, MID, GRID,
                           lighten, FONT_TITLE, FONT_LABEL, FONT_TICK)

font = setup()

fig, ax = plt.subplots(figsize=(3.8, 4.0))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

box_w, box_h = 7.0, 1.6
cx = 5.0  # center x

# --- Box positions (top to bottom) ---
y_sentence = 8.0
y_model = 5.0
y_vector = 2.0


def draw_box(ax, cx, cy, w, h, text, fontsize, text_color=DARK,
             facecolor='white', edgecolor=GRID, linewidth=1.2):
    x0 = cx - w / 2
    y0 = cy - h / 2
    box = mpatches.FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle="round,pad=0.3",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth,
        zorder=2)
    ax.add_patch(box)
    ax.text(cx, cy, text, ha='center', va='center',
            fontsize=fontsize, color=text_color, zorder=3)


# Box 1: Sentence (actual O*NET DWA)
draw_box(ax, cx, y_sentence, box_w, box_h,
         '"Analyze business or\nfinancial data."',
         fontsize=FONT_TITLE, text_color=DARK,
         facecolor='white', edgecolor=GRID)

# Box 2: Model
draw_box(ax, cx, y_model, box_w, box_h,
         'Sentence Embedding\nModel (e.g., MPNet)',
         fontsize=FONT_LABEL, text_color='white',
         facecolor=PRIMARY, edgecolor=PRIMARY, linewidth=1.5)

# Box 3: Vector — actual MPNet embedding values, mathtext rendering
vector_text = (r'$[\,-0.04,\;\; 0.06,\;\; -0.07,\;\; \ldots,\;\; -0.00\,]$'
               '\n768 dimensions')
draw_box(ax, cx, y_vector, box_w, box_h,
         vector_text,
         fontsize=FONT_TICK, text_color=DARK,
         facecolor=lighten(PRIMARY, 0.9), edgecolor=GRID)

# --- Arrows (start/end at box edges with shrink padding) ---
arrow_kw = dict(arrowstyle='->', color=MID, lw=1.5,
                shrinkA=5, shrinkB=5, zorder=1)

# Sentence → Model
ax.annotate('', xy=(cx, y_model + box_h / 2),
            xytext=(cx, y_sentence - box_h / 2),
            arrowprops=arrow_kw)

# Model → Vector
ax.annotate('', xy=(cx, y_vector + box_h / 2),
            xytext=(cx, y_model - box_h / 2),
            arrowprops=arrow_kw)

plt.savefig('figures/fig8_embedding_flowchart.png', dpi=300,
            bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print(f"Saved figures/fig8_embedding_flowchart.png  (font: {font})")

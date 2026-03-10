"""
Figure 8 — Embedding flowchart
Target: Slide 3 (left panel) — What is a Sentence Embedding?
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from figures.style import (setup, PRIMARY, DARK, MID, GRID,
                           lighten, FONT_TITLE, FONT_LABEL, FONT_TICK)
from src.task_space.data.onet import get_dwa_titles
from src.task_space.data.artifacts import get_embeddings
from src.task_space.domain import build_dwa_activity_domain

font = setup()

# === Load DWA embeddings from cache ===
domain = build_dwa_activity_domain()
dwa_titles = get_dwa_titles()
dwa_texts = [dwa_titles.get(aid, aid) for aid in domain.activity_ids]
embeddings = get_embeddings(dwa_texts, model='all-mpnet-base-v2')

# Find example sentence
EXAMPLE_SENTENCE = "Analyze business or financial data."
idx = dwa_texts.index(EXAMPLE_SENTENCE)
vec = embeddings[idx]
print(f"Example: {EXAMPLE_SENTENCE}")
print(f"  embedding shape: {vec.shape}, first 3: {vec[:3]}, last: {vec[-1]}")


def fmt(x):
    """Format float, collapsing -0.00 to 0.00."""
    s = f'{x:.2f}'
    return '0.00' if s == '-0.00' else s


vec_display = f'[{fmt(vec[0])},  {fmt(vec[1])},  {fmt(vec[2])},  ...,  {fmt(vec[-1])}]'
print(f"  display: {vec_display}")

# === Figure ===
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

# Box 3: Vector — live embedding values
draw_box(ax, cx, y_vector, box_w, box_h,
         vec_display + '\n768 dimensions',
         fontsize=FONT_TICK, text_color=DARK,
         facecolor=lighten(PRIMARY, 0.9), edgecolor=GRID)

# --- Arrows (start/end at box edges with shrink padding) ---
arrow_kw = dict(arrowstyle='->', color=MID, lw=1.5,
                shrinkA=8, shrinkB=8, zorder=1)

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

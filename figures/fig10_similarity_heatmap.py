"""
Figure 10 — Cosine similarity heatmap for 6 hand-picked DWAs.
Two tasks each from Quantitative, Healthcare, and Construction.
Matches the labeled examples and colors from fig3_task_scatter (slide 4).
Target: Slide 3 ("What is a Sentence Embedding?"), replacing word analogy figure.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics.pairwise import cosine_similarity

from figures.style import (setup, DARK, MID, GRID, PRIMARY, SECONDARY,
                           RED, FONT_TITLE, FONT_LABEL, FONT_TICK,
                           FONT_NOTE, add_subtitle, lighten)
from src.task_space.data.onet import get_dwa_titles
from src.task_space.data.artifacts import get_embeddings
from src.task_space.domain import build_dwa_activity_domain

font = setup()

# === Target tasks — 2 per domain, matching labeled DWAs on slide 4 scatter ===
TASKS = [
    'Calculate financial data.',
    'Balance receipts.',
    'Prescribe medications.',
    'Feed patients.',
    'Apply mortar.',
    'Cut glass.',
]

# Short display labels — full text for both axes
LABELS = [t.rstrip('.') for t in TASKS]
COL_LABELS = ['Fin. data', 'Receipts', 'Presc. meds', 'Feed pts.', 'Mortar', 'Cut glass']

# Domain grouping — colors match fig3_task_scatter CLUSTER_COLORS via style.py
DOMAINS = [
    ('Quantitative',  PRIMARY,   0, 1),
    ('Healthcare',    RED,       2, 3),
    ('Construction',  SECONDARY, 4, 5),
]

# === Load embeddings (from cache) ===
domain = build_dwa_activity_domain()
dwa_titles_dict = get_dwa_titles()
dwa_ids = domain.activity_ids
dwa_texts = [dwa_titles_dict.get(aid, aid) for aid in dwa_ids]
embeddings = get_embeddings(dwa_texts, model="all-mpnet-base-v2")

# Extract the 6 target task embeddings
title_to_idx = {t: i for i, t in enumerate(dwa_texts)}
indices = []
for t in TASKS:
    idx = title_to_idx.get(t)
    if idx is None:
        raise ValueError(f"Task not found: {t}")
    indices.append(idx)

task_emb = embeddings[indices]  # (6, 768)

# === Compute cosine similarity ===
sim_matrix = cosine_similarity(task_emb)  # (6, 6)

print("Cosine similarity matrix:")
for i, label in enumerate(LABELS):
    row = "  ".join(f"{sim_matrix[i, j]:.2f}" for j in range(len(LABELS)))
    print(f"  {label:30s}  {row}")

# === Plot ===
fig, ax = plt.subplots(figsize=(5.2, 5.2))

# Custom colormap: white (0) -> steel blue (1)
cmap = mcolors.LinearSegmentedColormap.from_list(
    'sim', ['#FFFFFF', lighten(PRIMARY, 0.5), PRIMARY], N=256
)

im = ax.imshow(sim_matrix, cmap=cmap, vmin=0, vmax=1, aspect='equal')

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.08, shrink=0.85)
cbar.set_label('Cosine similarity', fontsize=FONT_LABEL, color=MID)
cbar.ax.tick_params(labelsize=FONT_TICK, colors=MID)

# Annotate cells with similarity values — uniform font size
for i in range(len(LABELS)):
    for j in range(len(LABELS)):
        val = sim_matrix[i, j]
        text_color = '#FFFFFF' if val > 0.55 else DARK
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                fontsize=FONT_TITLE, color=text_color)

# Map task index to domain color
TASK_COLORS = {}
for domain_name, color, i_start, i_end in DOMAINS:
    for i in range(i_start, i_end + 1):
        TASK_COLORS[i] = color

# Axis labels — color by domain, columns on top
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')
ax.set_xticks(range(len(LABELS)))
ax.set_xticklabels(COL_LABELS, fontsize=FONT_LABEL, color=DARK,
                   rotation=45, ha='left', rotation_mode='anchor', fontweight='bold')
ax.set_yticks(range(len(LABELS)))
ax.set_yticklabels(LABELS, fontsize=FONT_LABEL, color=DARK, fontweight='bold')

# Color individual tick labels by domain
for i, label in enumerate(ax.get_yticklabels()):
    label.set_color(TASK_COLORS[i])
for i, label in enumerate(ax.get_xticklabels()):
    label.set_color(TASK_COLORS[i])

# Remove spines, add grid separators between domain groups
ax.tick_params(length=0)
for spine in ax.spines.values():
    spine.set_visible(False)

# Draw domain separator lines
for sep in [1.5, 3.5]:
    ax.axhline(sep, color=GRID, linewidth=1.2, zorder=2)
    ax.axvline(sep, color=GRID, linewidth=1.2, zorder=2)

# Subtle border around each diagonal 2×2 block
import matplotlib.patches as mpatches
for _, color, i_start, i_end in DOMAINS:
    rect = mpatches.FancyBboxPatch(
        (i_start - 0.5, i_start - 0.5), i_end - i_start + 1, i_end - i_start + 1,
        boxstyle='round,pad=0.02', linewidth=1.4, edgecolor=color,
        facecolor='none', alpha=0.7, zorder=3)
    ax.add_patch(rect)

# No colorbar needed — values annotated directly in cells

# Subtitle positioned relative to axes — stays close to matrix
ax.text(0.5, -0.10, '6 O*NET DWAs embedded with MPNet (768d)',
        transform=ax.transAxes, ha='center', fontsize=FONT_NOTE,
        color=MID, fontstyle='italic')

plt.tight_layout()
outpath = os.path.join(os.path.dirname(__file__), 'fig10_similarity_heatmap.png')
fig.savefig(outpath, dpi=300, bbox_inches='tight', pad_inches=0.1)

print(f"\nSaved: {outpath}")
plt.close()

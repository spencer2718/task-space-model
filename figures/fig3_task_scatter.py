"""
Figure 3 v9 — Task embeddings with Routine/Manual semantic axes
Hand-curated DWAs: 5 themes × ~5 dots × 1 label each.
Background cloud: all 2,087 DWAs projected live.
Target: Slide 4 ("Tasks in Semantic Space")
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

from figures.style import (setup, DARK, MID, GRID, PRIMARY, SECONDARY,
                           RED, ORANGE, GREEN, FONT_TICK, add_subtitle)
from src.task_space.data.onet import get_dwa_titles
from src.task_space.data.artifacts import get_embeddings
from src.task_space.domain import build_dwa_activity_domain

font = setup()

# === Theme colors ===
CLUSTER_COLORS = {
    'Healthcare':          RED,
    'Vehicle & Equipment': ORANGE,
    'Construction':        SECONDARY,
    'Quantitative':        PRIMARY,
    'Communication':       GREEN,
}

# === Hand-curated DWAs by theme ===
# Each entry: list of DWA titles (must match O*NET exactly, with trailing period)
CURATED = {
    'Healthcare': [
        'Feed patients.',
        'Give medications or immunizations.',
        'Position patients for treatment or examination.',
        'Move patients to or from treatment areas.',
        'Test patient vision.',
        'Prescribe medications.',
    ],
    'Vehicle & Equipment': [
        'Pilot aircraft.',
        'Operate forklifts or other loaders.',
        'Inspect motor vehicles.',
        'Clean vehicles or vehicle components.',
        'Assist passengers during vehicle boarding.',
        'Install vehicle parts or accessories.',
    ],
    'Construction': [
        'Cut glass.',
        'Apply mortar.',
        'Weld metal components.',
        'Spread concrete or other aggregate mixtures.',
        'Cut tile, stone, or other masonry materials.',
        'Dig holes or trenches.',
    ],
    'Quantitative': [
        'Balance receipts.',
        'Calculate financial data.',
        'Gather financial records.',
        'Examine financial records.',
        'Assess financial status of clients.',
        'Prepare operational budgets.',
    ],
    'Communication': [
        'Write grant proposals.',
        'Write reports or evaluations.',
        'Edit documents.',
        'Edit written materials.',
        'Type documents.',
        'Transcribe spoken or written information.',
    ],
}

# === Label config: 1 label per theme ===
LABELED = {
    'Feed patients.',
    'Prescribe medications.',
    'Inspect motor vehicles.',
    'Operate forklifts or other loaders.',
    'Cut glass.',
    'Apply mortar.',
    'Balance receipts.',
    'Calculate financial data.',
    'Write grant proposals.',
    'Transcribe spoken or written information.',
}

# Annotation offsets (dx, dy) in points
LABEL_OFFSETS = {
    'Feed patients.':         (-8, -12),
    'Prescribe medications.': (-10, -8),
    'Inspect motor vehicles.': (3, 9),
    'Operate forklifts or other loaders.': (0, -14),
    'Apply mortar.':          (10, 2),
    'Cut glass.':             (8, 5),
    'Balance receipts.':      (0, 10),
    'Calculate financial data.': (10, -6),
    'Write grant proposals.': (8, 8),
    'Transcribe spoken or written information.': (0, -11),
}

# Display-name overrides for long titles
DISPLAY_NAME = {
    'Operate forklifts or other loaders.': 'Operate forklifts',
    'Transcribe spoken or written information.': 'Transcribe information',
}

# Alignment overrides (bypass the offset-based ha logic)
HA_OVERRIDE = {
    'Inspect motor vehicles.': 'center',
}

# === Semantic anchor phrases ===
anchors = {
    'non_routine': "creative non-routine problem solving, judgment, and novel decision-making",
    'routine':     "routine repetitive procedural rule-following structured tasks",
    'cognitive':   "cognitive analytical intellectual knowledge-based information processing",
    'manual':      "manual physical hands-on bodily labor using tools and equipment",
}

# === Load embeddings and project ===
print("Loading DWA domain and embeddings...")
domain = build_dwa_activity_domain()
dwa_titles_dict = get_dwa_titles()
dwa_ids = domain.activity_ids
dwa_texts = [dwa_titles_dict.get(aid, aid) for aid in dwa_ids]
embeddings = get_embeddings(dwa_texts, model="all-mpnet-base-v2")
print(f"Loaded {len(dwa_ids)} DWAs, embeddings shape: {embeddings.shape}")

# Embed anchors
anchor_texts = list(anchors.values())
anchor_keys = list(anchors.keys())
anchor_emb = get_embeddings(anchor_texts, model="all-mpnet-base-v2")
anchor_dict = {k: anchor_emb[i:i+1] for i, k in enumerate(anchor_keys)}

# Project onto semantic axes
sim_nr = cos_sim(embeddings, anchor_dict['non_routine']).flatten()
sim_r  = cos_sim(embeddings, anchor_dict['routine']).flatten()
sim_c  = cos_sim(embeddings, anchor_dict['cognitive']).flatten()
sim_m  = cos_sim(embeddings, anchor_dict['manual']).flatten()

raw_x = sim_nr - sim_r
raw_y = sim_c - sim_m
all_x = (raw_x - raw_x.mean()) / raw_x.std() * 2.5
all_y = (raw_y - raw_y.mean()) / raw_y.std() * 2.5

# === Resolve curated DWAs against live data ===
title_to_idx = {t: i for i, t in enumerate(dwa_texts)}

selected = []
for theme, titles in CURATED.items():
    for title in titles:
        idx = title_to_idx.get(title)
        if idx is None:
            print(f"WARNING: '{title}' not found in DWA titles")
            continue
        selected.append({
            'dwa_id': dwa_ids[idx],
            'description': title,
            'cluster_name': theme,
            'x': float(all_x[idx]),
            'y': float(all_y[idx]),
            'labeled': title in LABELED,
        })

sel_df = pd.DataFrame(selected)
print(f"\nSelected {len(sel_df)} DWAs, {sel_df['labeled'].sum()} labeled")

for theme in CLUSTER_COLORS:
    group = sel_df[sel_df['cluster_name'] == theme]
    cx = group['x'].mean()
    cy = group['y'].mean()
    print(f"  {theme:<22s}  {len(group)} dots  centroid=({cx:+.2f}, {cy:+.2f})")
    for _, row in group.iterrows():
        marker = '  *' if row['labeled'] else '   '
        label = row['description'].rstrip('.')
        print(f"   {marker} ({row['x']:+.2f}, {row['y']:+.2f})  {label}")

# === Build figure ===
fig, ax = plt.subplots(figsize=(6.0, 4.0))

# Background cloud — ALL DWAs
ax.scatter(all_x, all_y, s=3, c='#E0E0E0', alpha=0.3,
           edgecolors='none', zorder=0, rasterized=True)

# Zero-crossing reference lines
ax.axhline(0, color=GRID, linewidth=0.8, zorder=0)
ax.axvline(0, color=GRID, linewidth=0.8, zorder=0)

# Selected DWAs by theme
for theme in CLUSTER_COLORS:
    group = sel_df[sel_df['cluster_name'] == theme]
    if group.empty:
        continue
    color = CLUSTER_COLORS[theme]
    ax.scatter(group['x'], group['y'], s=50, c=color,
               edgecolors='white', linewidths=0.5, alpha=0.9, zorder=2)

# Labels — 1 per theme, positioned via ax.annotate
for _, row in sel_df[sel_df['labeled']].iterrows():
    label = DISPLAY_NAME.get(row['description'], row['description'].rstrip('.'))
    offset = LABEL_OFFSETS.get(row['description'], (8, 5))
    color = CLUSTER_COLORS[row['cluster_name']]
    ax.annotate(label, xy=(row['x'], row['y']),
                xytext=offset, textcoords='offset points',
                fontsize=FONT_TICK, color=color, fontweight='bold',
                ha=HA_OVERRIDE.get(row['description'],
                   'center' if offset[0] == 0 else ('left' if offset[0] > 0 else 'right')),
                va='center', zorder=3)

# Legend — lower right
handles = [Line2D([0], [0], marker='o', color='w',
                  markerfacecolor=CLUSTER_COLORS[t], markersize=8, label=t)
           for t in CLUSTER_COLORS]
ax.legend(handles=handles, loc='upper left', fontsize=FONT_TICK, framealpha=0.9,
          edgecolor=GRID, fancybox=False, handletextpad=0.4)

# Axis labels
ax.set_xlabel('\u2190 Routine          Non-Routine \u2192',
              fontsize=11, color=MID, labelpad=8)
ax.set_ylabel('\u2190 Manual          Cognitive \u2192',
              fontsize=11, color=MID, labelpad=8)

# Chrome
ax.set_xlim(-5.5, 5.5)
ax.set_ylim(-6.5, 6.5)
ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color(GRID)
ax.spines['left'].set_color(GRID)

plt.tight_layout()
add_subtitle(fig, f'{len(sel_df)} selected DWAs from {len(dwa_ids):,} \u2014 projected onto interpretable semantic axes')
plt.savefig('figures/fig3_task_scatter.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"\nSaved figures/fig3_task_scatter.png  (font: {font})")

# Save CSV
out_df = sel_df[['dwa_id', 'description', 'cluster_name', 'x', 'y', 'labeled']]
out_df.columns = ['dwa_id', 'description', 'cluster_name', 'semantic_x', 'semantic_y', 'labeled']
out_df.to_csv('figures/fig3_selected_dwas.csv', index=False)
print(f"Saved figures/fig3_selected_dwas.csv  ({len(out_df)} DWAs)")

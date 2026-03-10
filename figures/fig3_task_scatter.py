"""
Figure 3 v8 — Task embeddings with Routine/Manual semantic axes
Keyword-only clustering: match DWA titles against theme patterns.
5 themes × 6 dots × 2 labels, adjustText.
Target: Slide 4 ("Tasks in Semantic Space")
"""
import sys, os, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from scipy.spatial import cKDTree
from adjustText import adjust_text

from figures.style import (setup, DARK, MID, GRID, PRIMARY, SECONDARY,
                           RED, ORANGE, GREEN, FONT_TICK, add_subtitle)
from src.task_space.data.onet import get_dwa_titles
from src.task_space.data.artifacts import get_embeddings
from src.task_space.domain import build_dwa_activity_domain

font = setup()

CLUSTER_COLORS = {
    'Healthcare':          RED,
    'Vehicle & Equipment': ORANGE,
    'Construction':        SECONDARY,
    'Quantitative':        PRIMARY,
    'Communication':       GREEN,
}

# === Semantic anchor phrases ===
anchors = {
    'non_routine': "creative non-routine problem solving, judgment, and novel decision-making",
    'routine':     "routine repetitive procedural rule-following structured tasks",
    'cognitive':   "cognitive analytical intellectual knowledge-based information processing",
    'manual':      "manual physical hands-on bodily labor using tools and equipment",
}

# === Theme definitions: keywords matched against DWA titles ===
THEME_PATTERNS = {
    'Healthcare':          ['patient', 'medical', 'health', 'nurs', 'treat', 'diagnos',
                            'immuniz', 'prescri', 'symptom', 'therap'],
    'Vehicle & Equipment': ['vehicle', 'drive', 'truck', 'aircraft', 'pilot', 'forklift',
                            'engine', 'tire', 'cargo', r'operat.*crane', 'tow '],
    'Construction':        ['weld', 'masonry', 'concrete', 'plumb', 'mortar', 'excavat',
                            'trench', 'scaffold', 'roofing', 'drywall', 'flooring', 'lumber'],
    'Quantitative':        ['financial', 'statistic', 'budget', 'forecast', 'audit',
                            'tax', 'accounting', 'actuari'],
    'Communication':       ['edit document', 'edit written', r'write ', 'proofread',
                            'transcri', 'publish', 'draft ', 'grant proposal'],
}


def match_theme(title):
    title_lower = title.lower()
    for theme, patterns in THEME_PATTERNS.items():
        if any(re.search(p, title_lower) for p in patterns):
            return theme
    return None


# === Load embeddings ===
print("Loading DWA domain and embeddings...")
domain = build_dwa_activity_domain()
dwa_titles_dict = get_dwa_titles()
dwa_ids = domain.activity_ids
dwa_texts = [dwa_titles_dict.get(aid, aid) for aid in dwa_ids]
titles_clean = [t.rstrip('.') for t in dwa_texts]
embeddings = get_embeddings(dwa_texts, model="all-mpnet-base-v2")
print(f"Loaded {len(dwa_ids)} DWAs, embeddings shape: {embeddings.shape}")

# === Match themes ===
themes = [match_theme(t) for t in dwa_texts]

print("\nTheme match counts:")
for theme in CLUSTER_COLORS:
    idxs = [i for i, t in enumerate(themes) if t == theme]
    print(f"  {theme:<22s}  {len(idxs)}")
    if len(idxs) < 6:
        print(f"    WARNING: < 6 matches. Titles:")
        for i in idxs:
            print(f"      {titles_clean[i]}")

# === Embed anchors ===
anchor_texts = list(anchors.values())
anchor_keys = list(anchors.keys())
anchor_emb = get_embeddings(anchor_texts, model="all-mpnet-base-v2")
anchor_dict = {k: anchor_emb[i:i+1] for i, k in enumerate(anchor_keys)}

# === Project onto semantic axes ===
sim_nr = cos_sim(embeddings, anchor_dict['non_routine']).flatten()
sim_r  = cos_sim(embeddings, anchor_dict['routine']).flatten()
sim_c  = cos_sim(embeddings, anchor_dict['cognitive']).flatten()
sim_m  = cos_sim(embeddings, anchor_dict['manual']).flatten()

raw_x = sim_nr - sim_r
raw_y = sim_c - sim_m

print(f"\nRaw X: mean={raw_x.mean():.4f}, std={raw_x.std():.4f}")
print(f"Raw Y: mean={raw_y.mean():.4f}, std={raw_y.std():.4f}")

if raw_x.std() < 0.01 or raw_y.std() < 0.01:
    print("STOP: Axis std < 0.01 — anchors not discriminating")
    sys.exit(1)

all_x = (raw_x - raw_x.mean()) / raw_x.std() * 2.5
all_y = (raw_y - raw_y.mean()) / raw_y.std() * 2.5


# === Density-based selection ===
def find_densest_group(idxs, all_x, all_y, n=6):
    """Find n points in the densest region using k-NN seed."""
    if len(idxs) <= n:
        return list(idxs)

    xs = np.array([all_x[i] for i in idxs])
    ys = np.array([all_y[i] for i in idxs])

    tree = cKDTree(np.column_stack([xs, ys]))
    k = min(n - 1, len(idxs) - 1)
    dists, _ = tree.query(np.column_stack([xs, ys]), k=k + 1)
    density = dists[:, 1:].mean(axis=1)

    seed = np.argmin(density)
    _, nn_idxs = tree.query(np.column_stack([xs, ys])[seed:seed + 1], k=n)
    return [idxs[i] for i in nn_idxs[0]]


def pick_labels(group, n=2, max_len=28):
    """Pick the n shortest-titled DWAs that fit within max_len."""
    candidates = [(i, len(titles_clean[i])) for i in group if len(titles_clean[i]) <= max_len]
    candidates.sort(key=lambda x: x[1])
    return set(c[0] for c in candidates[:n])


selected = []
print("\n=== Selected groups ===")
for theme in CLUSTER_COLORS:
    idxs = [i for i, t in enumerate(themes) if t == theme]
    if len(idxs) < 3:
        print(f"WARNING: {theme} has only {len(idxs)} matches — skipping")
        continue

    n_group = min(6, len(idxs))
    group = find_densest_group(idxs, all_x, all_y, n=n_group)
    labeled_set = pick_labels(group, n=2)

    cx = np.mean([all_x[i] for i in group])
    cy = np.mean([all_y[i] for i in group])

    print(f"\n{theme} ({len(group)} dots, centroid=({cx:+.2f}, {cy:+.2f})):")
    for idx in group:
        is_labeled = idx in labeled_set
        marker = '  *' if is_labeled else '   '
        print(f" {marker} ({all_x[idx]:+.2f}, {all_y[idx]:+.2f})  [{len(titles_clean[idx]):2d}] {titles_clean[idx]}")
        selected.append({
            'dwa_id': dwa_ids[idx],
            'description': dwa_texts[idx],
            'cluster_name': theme,
            'x': float(all_x[idx]),
            'y': float(all_y[idx]),
            'labeled': is_labeled,
        })

sel_df = pd.DataFrame(selected)
n_labeled = sel_df['labeled'].sum()
print(f"\nSelected {len(sel_df)} DWAs total, {n_labeled} labeled")

# Print final label strings
print("\nFinal labels:")
for _, row in sel_df[sel_df['labeled']].iterrows():
    label = row['description'].rstrip('.')
    print(f"  [{len(label):2d}] {label}")

# === Build figure ===
fig, ax = plt.subplots(figsize=(6.0, 4.0))

# Background cloud — ALL 2,087 DWAs
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

# Labels via adjustText
texts = []
for _, row in sel_df.iterrows():
    if row['labeled']:
        t = ax.text(row['x'], row['y'], row['description'].rstrip('.'),
                    fontsize=FONT_TICK, color=CLUSTER_COLORS[row['cluster_name']],
                    zorder=3)
        texts.append(t)

adjust_text(texts, ax=ax,
            arrowprops=dict(arrowstyle='-', color=GRID, lw=0.5),
            force_text=(0.3, 0.5),
            force_points=(0.5, 0.5),
            expand_text=(1.1, 1.2),
            lim=1000)

# Post-adjustText: flip any label that lands in legend region (lower-right)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
for t in texts:
    bbox = t.get_window_extent(renderer)
    data_bbox = ax.transData.inverted().transform(bbox)
    # If label center is in lower-right quadrant (legend area)
    cx_label = (data_bbox[0, 0] + data_bbox[1, 0]) / 2
    cy_label = (data_bbox[0, 1] + data_bbox[1, 1]) / 2
    if cx_label > 2.0 and cy_label < -2.0:
        # Move label to opposite side of its dot
        pos = t.get_position()
        t.set_position((pos[0] - 2.0, pos[1] + 1.0))

# Legend — lower right
handles = [Line2D([0], [0], marker='o', color='w',
                  markerfacecolor=CLUSTER_COLORS[t], markersize=8, label=t)
           for t in CLUSTER_COLORS]
ax.legend(handles=handles, loc='lower right', fontsize=FONT_TICK, framealpha=0.9,
          edgecolor=GRID, fancybox=False, handletextpad=0.4)

# Axis labels
ax.set_xlabel('\u2190 Routine          Non-Routine \u2192',
              fontsize=11, color=MID, labelpad=8)
ax.set_ylabel('\u2190 Manual          Cognitive \u2192',
              fontsize=11, color=MID, labelpad=8)

# Chrome
ax.set_xlim(-5.5, 5.5)
ax.set_ylim(-5.5, 5.5)
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

"""
Figure 3 v6 — Task embeddings with Routine/Manual semantic axes
5 themes × 6 dots × 2 labels, adjustText for collision-free placement.
Target: Slide 4 ("Tasks in Semantic Space")
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from scipy.spatial import cKDTree
from adjustText import adjust_text

from figures.style import (setup, DARK, MID, GRID, PRIMARY, SECONDARY,
                           RED, ORANGE, GREEN, FONT_LABEL, FONT_TICK, add_subtitle)
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

# === Stage 1: Narrowed GWA-to-theme mapping ===
GWA_TO_THEME = {
    'Assisting and Caring for Others': 'Healthcare',
    'Performing General Physical Activities': 'Construction',
    'Repairing and Maintaining Mechanical Equipment': 'Vehicle & Equipment',
    'Operating Vehicles, Mechanized Devices, or Equipment': 'Vehicle & Equipment',
    'Analyzing Data or Information': 'Quantitative',
    'Estimating the Quantifiable Characteristics of Products, Events, or Information': 'Quantitative',
    'Communicating with People Outside the Organization': 'Communication',
    'Documenting/Recording Information': 'Communication',
}

# === Stage 2: Keyword confirmation ===
THEME_KEYWORDS = {
    'Healthcare':          ['patient', 'medical', 'health', 'treat', 'diagnos', 'nurs',
                            'therap', 'immuniz', 'prescri', 'symptom'],
    'Vehicle & Equipment': ['vehicle', 'drive', 'truck', 'aircraft', 'pilot', 'forklift',
                            'engine', 'mechanic', 'tire', 'cargo', 'equipment'],
    'Construction':        ['construct', 'build', 'weld', 'masonry', 'concrete', 'plumb',
                            'excavat', 'trench', 'scaffold', 'install', 'roofing', 'mortar',
                            'lumber'],
    'Quantitative':        ['financial', 'data', 'statistic', 'calculat', 'budget', 'quantit',
                            'forecast', 'analyz', 'audit', 'account'],
    'Communication':       ['document', 'write', 'edit', 'report', 'present', 'correspond',
                            'publish', 'transcri', 'proofread'],
}


def passes_keyword_filter(title, theme):
    title_lower = title.lower()
    return any(kw in title_lower for kw in THEME_KEYWORDS[theme])


# === Load embeddings ===
print("Loading DWA domain and embeddings...")
domain = build_dwa_activity_domain()
dwa_titles = get_dwa_titles()
dwa_ids = domain.activity_ids
dwa_texts = [dwa_titles.get(aid, aid) for aid in dwa_ids]
embeddings = get_embeddings(dwa_texts, model="all-mpnet-base-v2")
print(f"Loaded {len(dwa_ids)} DWAs, embeddings shape: {embeddings.shape}")

# === Load GWA metadata ===
dwa_meta = pd.read_excel('data/onet/db_30_0_excel/DWA Reference.xlsx')
dwa_meta = dwa_meta[['DWA ID', 'Element Name']].drop_duplicates(subset='DWA ID').sort_values('DWA ID')
dwa_id_to_gwa = dict(zip(dwa_meta['DWA ID'], dwa_meta['Element Name']))
gwa_names = [dwa_id_to_gwa.get(aid, '') for aid in dwa_ids]

# Two-stage theme assignment
themes = []
for i, gwa in enumerate(gwa_names):
    candidate_theme = GWA_TO_THEME.get(gwa, None)
    if candidate_theme and passes_keyword_filter(dwa_texts[i], candidate_theme):
        themes.append(candidate_theme)
    else:
        themes.append(None)

print("\nTheme pool sizes (after GWA + keyword filter):")
for theme in CLUSTER_COLORS:
    n = sum(1 for t in themes if t == theme)
    print(f"  {theme:<22s}  {n}")

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


# === Density-based selection: densest group of n per theme ===
def find_densest_group(idxs, all_x, all_y, n=6):
    """Find n points in the densest region using k-nearest neighbors."""
    xs = np.array([all_x[i] for i in idxs])
    ys = np.array([all_y[i] for i in idxs])

    tree = cKDTree(np.column_stack([xs, ys]))
    k = min(n - 1, len(idxs) - 1)
    dists, _ = tree.query(np.column_stack([xs, ys]), k=k + 1)
    density = dists[:, 1:].mean(axis=1)

    # Seed: the densest point and its k nearest neighbors
    seed = np.argmin(density)
    _, nn_idxs = tree.query(np.column_stack([xs, ys])[seed:seed + 1], k=min(n, len(idxs)))
    group = [idxs[i] for i in nn_idxs[0]]
    return group


def pick_labels(group_idxs, n_labels=2):
    """Pick the n shortest-titled DWAs for labeling."""
    by_len = sorted(group_idxs, key=lambda i: len(dwa_texts[i]))
    return set(by_len[:n_labels])


def short_label(text, max_len=28):
    """Trim DWA text to a short label."""
    text = text.rstrip('.')
    if len(text) <= max_len:
        return text
    cut = text[:max_len].rfind(' ')
    return text[:cut] + '...' if cut > 10 else text[:max_len - 3] + '...'


selected = []
print("\n=== Selected groups ===")
for theme in CLUSTER_COLORS:
    idxs = [i for i, t in enumerate(themes) if t == theme]
    n_available = len(idxs)
    n_group = min(6, n_available)
    if n_available < 3:
        print(f"WARNING: {theme} has only {n_available} DWAs")
        continue
    if n_available < 6:
        print(f"NOTE: {theme} has only {n_available} DWAs, using all")

    group = find_densest_group(idxs, all_x, all_y, n=n_group)
    labeled_set = pick_labels(group, n_labels=2)

    print(f"\n{theme} ({n_group} dots, 2 labeled):")
    for idx in group:
        is_labeled = idx in labeled_set
        marker = '  *' if is_labeled else '   '
        print(f" {marker} ({all_x[idx]:+.2f}, {all_y[idx]:+.2f})  {dwa_texts[idx]}")
        selected.append({
            'dwa_id': dwa_ids[idx],
            'description': dwa_texts[idx],
            'cluster_name': theme,
            'x': float(all_x[idx]),
            'y': float(all_y[idx]),
            'labeled': is_labeled,
        })

sel_df = pd.DataFrame(selected)
print(f"\nSelected {len(sel_df)} DWAs total, {sel_df['labeled'].sum()} labeled")

# === Build figure ===
fig, ax = plt.subplots(figsize=(6.0, 4.0))

# Background cloud
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
        t = ax.text(row['x'], row['y'], short_label(row['description']),
                    fontsize=FONT_LABEL, color=CLUSTER_COLORS[row['cluster_name']],
                    zorder=3)
        texts.append(t)

adjust_text(texts, ax=ax,
            arrowprops=dict(arrowstyle='-', color=GRID, lw=0.5),
            force_text=(0.5, 0.8),
            force_points=(0.8, 0.8),
            expand_text=(1.2, 1.4),
            lim=500)

# Legend — lower right
handles = [Line2D([0], [0], marker='o', color='w',
                  markerfacecolor=CLUSTER_COLORS[t], markersize=8, label=t)
           for t in CLUSTER_COLORS]
ax.legend(handles=handles, loc='lower right', fontsize=FONT_TICK, framealpha=0.9,
          edgecolor=GRID, fancybox=False, handletextpad=0.4)

# Axis labels
ax.set_xlabel('← Routine          Non-Routine →',
              fontsize=11, color=MID, labelpad=8)
ax.set_ylabel('← Manual          Cognitive →',
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

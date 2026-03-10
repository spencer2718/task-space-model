"""
Figure 3 v4 — Task embeddings with Routine/Manual semantic axes
Density-based DWA selection: tightest 3-DWA cluster per theme.
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
from itertools import combinations

from figures.style import (setup, DARK, MID, GRID, PRIMARY, SECONDARY,
                           RED, ORANGE, GREEN, PURPLE, FONT_TICK, add_subtitle)
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
    'Technology':          PURPLE,
}

# === Semantic anchor phrases ===
anchors = {
    'non_routine': "creative non-routine problem solving, judgment, and novel decision-making",
    'routine':     "routine repetitive procedural rule-following structured tasks",
    'cognitive':   "cognitive analytical intellectual knowledge-based information processing",
    'manual':      "manual physical hands-on bodily labor using tools and equipment",
}

# === GWA-to-theme mapping ===
GWA_TO_THEME = {
    'Assisting and Caring for Others': 'Healthcare',
    'Performing for or Working Directly with the Public': 'Healthcare',
    'Handling and Moving Objects': 'Construction',
    'Performing General Physical Activities': 'Construction',
    'Repairing and Maintaining Mechanical Equipment': 'Vehicle & Equipment',
    'Operating Vehicles, Mechanized Devices, or Equipment': 'Vehicle & Equipment',
    'Controlling Machines and Processes': 'Vehicle & Equipment',
    'Analyzing Data or Information': 'Quantitative',
    'Processing Information': 'Quantitative',
    'Estimating the Quantifiable Characteristics of Products, Events, or Information': 'Quantitative',
    'Communicating with Supervisors, Peers, or Subordinates': 'Communication',
    'Communicating with People Outside the Organization': 'Communication',
    'Documenting/Recording Information': 'Communication',
    'Interacting With Computers': 'Technology',
    'Thinking Creatively': 'Technology',
    'Updating and Using Relevant Knowledge': 'Technology',
}

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
# Align to domain order
dwa_id_to_gwa = dict(zip(dwa_meta['DWA ID'], dwa_meta['Element Name']))
gwa_names = [dwa_id_to_gwa.get(aid, '') for aid in dwa_ids]
themes = [GWA_TO_THEME.get(g, None) for g in gwa_names]

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

raw_x = sim_nr - sim_r   # routine (left) → non-routine (right)
raw_y = sim_c - sim_m    # manual (bottom) → cognitive (top)

print(f"Raw X: mean={raw_x.mean():.4f}, std={raw_x.std():.4f}")
print(f"Raw Y: mean={raw_y.mean():.4f}, std={raw_y.std():.4f}")

if raw_x.std() < 0.01 or raw_y.std() < 0.01:
    print("STOP: Axis std < 0.01 — anchors not discriminating")
    sys.exit(1)

# Rescale to ~±5
all_x = (raw_x - raw_x.mean()) / raw_x.std() * 2.5
all_y = (raw_y - raw_y.mean()) / raw_y.std() * 2.5


# === Density-based selection: tightest triple per theme ===
def find_densest_triple(idxs, all_x, all_y, max_search=30):
    """Find 3 points with minimum total pairwise 2D distance."""
    xs = np.array([all_x[i] for i in idxs])
    ys = np.array([all_y[i] for i in idxs])

    # For large groups, narrow search to densest region
    if len(idxs) > 50:
        tree = cKDTree(np.column_stack([xs, ys]))
        k = min(10, len(idxs) - 1)
        dists, _ = tree.query(np.column_stack([xs, ys]), k=k + 1)
        density = dists[:, 1:].mean(axis=1)
        top = np.argsort(density)[:max_search]
        search = [idxs[i] for i in top]
    else:
        search = list(idxs)

    best_dist = float('inf')
    best = None
    texts = [dwa_texts[i] for i in search]
    for combo in combinations(range(len(search)), 3):
        i, j, k = combo
        # Text diversity: require different first words
        first_words = {texts[i].split()[0], texts[j].split()[0], texts[k].split()[0]}
        if len(first_words) < 2:
            continue
        ii, jj, kk = search[i], search[j], search[k]
        d = (np.hypot(all_x[ii] - all_x[jj], all_y[ii] - all_y[jj]) +
             np.hypot(all_x[ii] - all_x[kk], all_y[ii] - all_y[kk]) +
             np.hypot(all_x[jj] - all_x[kk], all_y[jj] - all_y[kk]))
        if d < best_dist:
            best_dist = d
            best = (ii, jj, kk)
    return best, best_dist


selected = []
for theme in CLUSTER_COLORS:
    idxs = [i for i, t in enumerate(themes) if t == theme]
    if len(idxs) < 3:
        print(f"WARNING: {theme} has only {len(idxs)} DWAs")
        continue
    triple, dist = find_densest_triple(idxs, all_x, all_y)
    print(f"{theme}: dist={dist:.3f}")
    for idx in triple:
        print(f"  ({all_x[idx]:+.2f}, {all_y[idx]:+.2f})  {dwa_texts[idx]}")
        selected.append({
            'dwa_id': dwa_ids[idx],
            'description': dwa_texts[idx],
            'cluster_name': theme,
            'x': float(all_x[idx]),
            'y': float(all_y[idx]),
        })

sel_df = pd.DataFrame(selected)
print(f"\nSelected {len(sel_df)} DWAs total")


# === Short labels ===
def short_label(text, max_len=25):
    """Trim DWA text to a short label."""
    text = text.rstrip('.')
    if len(text) <= max_len:
        return text
    cut = text[:max_len].rfind(' ')
    return text[:cut] + '...' if cut > 10 else text[:max_len - 3] + '...'


# === Collision-aware label placement ===
# Approximate points-per-data-unit for the figure
fig_temp = plt.figure(figsize=(6.0, 4.0))
ax_temp = fig_temp.add_subplot(111)
ax_temp.set_xlim(-5.5, 5.5)
ax_temp.set_ylim(-5.5, 5.5)
dpi = fig_temp.dpi
ppdu_x = (6.0 * dpi) / 11.0
ppdu_y = (4.0 * dpi) / 11.0
plt.close(fig_temp)

label_positions = []
for _, row in sel_df.iterrows():
    x, y = row['x'], row['y']
    if x > 0:
        x_off, ha = 10, 'left'
    else:
        x_off, ha = -10, 'right'
    label_positions.append({
        'x': x, 'y': y, 'x_off': x_off, 'y_off': 0, 'ha': ha,
        'label': short_label(row['description']),
        'color': CLUSTER_COLORS[row['cluster_name']],
    })

# Collision resolution — bump overlapping labels vertically
for i in range(len(label_positions)):
    for j in range(i + 1, len(label_positions)):
        li, lj = label_positions[i], label_positions[j]
        dx = abs((li['x'] + li['x_off'] / ppdu_x) - (lj['x'] + lj['x_off'] / ppdu_x))
        dy = abs((li['y'] + li['y_off'] / ppdu_y) - (lj['y'] + lj['y_off'] / ppdu_y))
        if dx < 2.0 and dy < 0.6:
            li['y_off'] += 8
            lj['y_off'] -= 8

# Legend collision: if label lands in upper-left quadrant, flip to other side
for lp in label_positions:
    label_x = lp['x'] + lp['x_off'] / ppdu_x
    label_y = lp['y'] + lp['y_off'] / ppdu_y
    if label_x < -2.0 and label_y > 2.5:
        lp['x_off'] = 10
        lp['ha'] = 'left'

print("\nLabel positions:")
for lp in label_positions:
    print(f"  {lp['label']:<25s}  x_off={lp['x_off']:+d}  y_off={lp['y_off']:+d}  ha={lp['ha']}")

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

# Labels for all 18 selected DWAs
for lp in label_positions:
    ax.annotate(lp['label'], xy=(lp['x'], lp['y']), xycoords='data',
                xytext=(lp['x_off'], lp['y_off']), textcoords='offset points',
                fontsize=9, color=lp['color'], ha=lp['ha'], va='center',
                zorder=3)

# Legend — upper left
handles = [Line2D([0], [0], marker='o', color='w',
                  markerfacecolor=CLUSTER_COLORS[t], markersize=8, label=t)
           for t in CLUSTER_COLORS]
ax.legend(handles=handles, loc='upper left', fontsize=FONT_TICK, framealpha=0.9,
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
add_subtitle(fig, f'{len(sel_df)} selected DWAs from {len(dwa_ids):,} — projected onto interpretable semantic axes')
plt.savefig('figures/fig3_task_scatter.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"\nSaved figures/fig3_task_scatter.png  (font: {font})")

# Save CSV
out_df = sel_df[['dwa_id', 'description', 'cluster_name', 'x', 'y']]
out_df.columns = ['dwa_id', 'description', 'cluster_name', 'semantic_x', 'semantic_y']
out_df.to_csv('figures/fig3_selected_dwas.csv', index=False)
print(f"Saved figures/fig3_selected_dwas.csv  ({len(out_df)} DWAs)")

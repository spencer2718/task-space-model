"""
Figure 3 v3 — Task embeddings with Routine/Manual semantic axes + legend
Target: Slide 3 ("Core Idea: Tasks in Semantic Space")
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

from figures.style import setup, DARK, MID, GRID
from src.task_space.data.onet import get_dwa_titles
from src.task_space.data.artifacts import get_embeddings
from src.task_space.domain import build_dwa_activity_domain

font = setup()

CLUSTER_COLORS = {
    'Healthcare':          '#C75C5C',
    'Vehicle & Equipment': '#D4845A',
    'Construction':        '#A0937D',
    'Quantitative':        '#4A7FB5',
    'Communication':       '#6B9E78',
    'Technology':          '#8B6BAE',
}

# === Semantic anchor phrases ===
anchors = {
    'non_routine': "creative non-routine problem solving, judgment, and novel decision-making",
    'routine':     "routine repetitive procedural rule-following structured tasks",
    'cognitive':   "cognitive analytical intellectual knowledge-based information processing",
    'manual':      "manual physical hands-on bodily labor using tools and equipment",
}

# === Hand-selected DWAs ===
SELECTED_DWAS = {
    'Healthcare': [
        'Feed patients.',
        'Immunize patients.',
        'Prescribe medications.',
        'Test patient vision.',
        'Administer first aid.',
        'Treat medical emergencies.',
    ],
    'Vehicle & Equipment': [
        'Drive trucks or truck-mounted equipment.',
        'Pilot aircraft.',
        'Secure cargo.',
        'Operate forklifts or other loaders.',
        'Load shipments, belongings, or materials.',
        'Repair tires.',
    ],
    'Construction': [
        'Apply mortar.',
        'Cut glass.',
        'Dig holes or trenches.',
        'Weld metal components.',
        'Align masonry materials.',
        'Assemble temporary equipment or structures.',
    ],
    'Quantitative': [
        'Calculate tax information.',
        'Balance receipts.',
        'Prepare operational budgets.',
        'Calculate financial data.',
    ],
    'Communication': [
        'Edit documents.',
        'Type documents.',
        'Edit written materials.',
        'Write grant proposals.',
    ],
    'Technology': [
        'Install computer software.',
        'Configure computer networks.',
        'Design software applications.',
        'Test software performance.',
    ],
}

LABELED = {
    'Healthcare':          ['Prescribe medications', 'Immunize patients',
                            'Treat emergencies'],
    'Vehicle & Equipment': ['Pilot aircraft', 'Drive trucks',
                            'Operate forklifts'],
    'Construction':        ['Weld metal components', 'Dig holes or trenches'],
    'Quantitative':        ['Calculate tax information', 'Balance receipts'],
    'Communication':       ['Edit documents', 'Write grant proposals'],
    'Technology':          ['Design software applications',
                            'Configure computer networks'],
}

LABEL_OVERRIDES = {
    'Drive trucks or truck-mounted equipment.': 'Drive trucks',
    'Operate forklifts or other loaders.': 'Operate forklifts',
    'Load shipments, belongings, or materials.': 'Load shipments',
    'Assemble temporary equipment or structures.': 'Assemble structures',
    'Treat medical emergencies.': 'Treat emergencies',
}


def short_label(text):
    if text in LABEL_OVERRIDES:
        return LABEL_OVERRIDES[text]
    return text.rstrip('.')


# === Load embeddings ===
print("Loading DWA domain and embeddings...")
domain = build_dwa_activity_domain()
dwa_titles = get_dwa_titles()
dwa_ids = domain.activity_ids
dwa_texts = [dwa_titles.get(aid, aid) for aid in dwa_ids]
embeddings = get_embeddings(dwa_texts, model="all-mpnet-base-v2")
print(f"Loaded {len(dwa_ids)} DWAs, embeddings shape: {embeddings.shape}")

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

# Stop condition check
if raw_x.std() < 0.01 or raw_y.std() < 0.01:
    print("STOP: Axis std < 0.01 — anchors not discriminating")
    sys.exit(1)

# Rescale to ~±5
all_x = (raw_x - raw_x.mean()) / raw_x.std() * 2.5
all_y = (raw_y - raw_y.mean()) / raw_y.std() * 2.5

# === Match selected DWAs ===
text_to_idx = {t: i for i, t in enumerate(dwa_texts)}
selected = []
for theme, desc_list in SELECTED_DWAS.items():
    for desc in desc_list:
        if desc in text_to_idx:
            idx = text_to_idx[desc]
        else:
            d_lower = desc.lower()[:20]
            matches = [(i, t) for i, t in enumerate(dwa_texts)
                       if d_lower in t.lower()]
            if matches:
                idx = matches[0][0]
            else:
                print(f"  WARNING: '{desc}' not found, skipping")
                continue
        selected.append({
            'dwa_id': dwa_ids[idx],
            'description': dwa_texts[idx],
            'cluster_name': theme,
            'x': float(all_x[idx]),
            'y': float(all_y[idx]),
        })

sel_df = pd.DataFrame(selected).drop_duplicates(subset='dwa_id', keep='first')

# === Print cluster centroids ===
print("\nCluster centroids (scaled axes):")
print(f"  {'Cluster':<22s}  {'x (rtn→nr)':>10s}  {'y (man→cog)':>10s}  n")
for theme in CLUSTER_COLORS:
    g = sel_df[sel_df['cluster_name'] == theme]
    if g.empty:
        continue
    print(f"  {theme:<22s}  {g['x'].mean():>10.2f}  {g['y'].mean():>10.2f}  {len(g)}")

# === Compute label offsets based on actual positions ===
# Build labeled points list with coordinates
label_points = []
for theme in CLUSTER_COLORS:
    group = sel_df[sel_df['cluster_name'] == theme]
    labeled_set = set(LABELED.get(theme, []))
    for _, row in group.iterrows():
        sl = short_label(row['description'])
        if sl in labeled_set:
            label_points.append({
                'label': sl, 'x': row['x'], 'y': row['y'],
                'theme': theme,
            })

lp_df = pd.DataFrame(label_points)

# Auto-compute offsets: right half → label right, left half → label left
LABEL_OFFSETS = {}
for _, lp in lp_df.iterrows():
    if lp['x'] > 0:
        x_off, ha = 10, 'left'
    else:
        x_off, ha = -10, 'right'
    y_off = 0
    LABEL_OFFSETS[lp['label']] = [x_off, y_off, ha]

# Collision resolution: check pairs and bump vertically
fig_temp = plt.figure(figsize=(6.0, 4.2))
ax_temp = fig_temp.add_subplot(111)
ax_temp.set_xlim(-5.5, 5.5)
ax_temp.set_ylim(-5.5, 5.5)
dpi = fig_temp.dpi
# Approximate points-per-data-unit
ppdu_x = (6.0 * dpi) / 11.0  # 11 data units across
ppdu_y = (4.2 * dpi) / 11.0
plt.close(fig_temp)

for i, lp1 in lp_df.iterrows():
    for j, lp2 in lp_df.iterrows():
        if j <= i:
            continue
        o1 = LABEL_OFFSETS[lp1['label']]
        o2 = LABEL_OFFSETS[lp2['label']]
        # Label anchor in points
        ax1 = lp1['x'] * ppdu_x + o1[0]
        ay1 = lp1['y'] * ppdu_y + o1[1]
        ax2 = lp2['x'] * ppdu_x + o2[0]
        ay2 = lp2['y'] * ppdu_y + o2[1]
        dist = np.sqrt((ax1 - ax2)**2 + (ay1 - ay2)**2)
        if dist < 25:
            # Bump one up and one down
            o1[1] += 10
            o2[1] -= 10

# Convert to tuples
LABEL_OFFSETS = {k: tuple(v) for k, v in LABEL_OFFSETS.items()}

# Manual overrides for known tight spots
LABEL_OFFSETS['Pilot aircraft'] = (-10, 8, 'right')
LABEL_OFFSETS['Weld metal components'] = (10, -10, 'left')
LABEL_OFFSETS['Dig holes or trenches'] = (10, 8, 'left')
LABEL_OFFSETS['Drive trucks'] = (10, -8, 'left')
LABEL_OFFSETS['Treat emergencies'] = (10, -8, 'left')
LABEL_OFFSETS['Balance receipts'] = (10, 8, 'left')
LABEL_OFFSETS['Configure computer networks'] = (10, -8, 'left')
print("\nLabel offsets:")
for k, v in LABEL_OFFSETS.items():
    print(f"  {k:<35s}  {v}")

# === Build figure ===
fig, ax = plt.subplots(figsize=(6.0, 4.2))

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

    labeled_set = set(LABELED.get(theme, []))
    for _, row in group.iterrows():
        sl = short_label(row['description'])
        if sl not in labeled_set:
            continue
        offs = LABEL_OFFSETS.get(sl, (10, 0, 'left'))
        ax.annotate(sl, xy=(row['x'], row['y']), xycoords='data',
                    xytext=(offs[0], offs[1]), textcoords='offset points',
                    fontsize=9, color=color, ha=offs[2], va='center',
                    zorder=3)

# Legend
handles = [Line2D([0], [0], marker='o', color='w',
                  markerfacecolor=CLUSTER_COLORS[t], markersize=8, label=t)
           for t in CLUSTER_COLORS]
ax.legend(handles=handles, loc='lower right', fontsize=8.5, framealpha=0.9,
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
fig.text(0.5, -0.03,
         '30 selected DWAs from 2,087 — projected onto interpretable semantic axes',
         ha='center', fontsize=8, color=MID, fontstyle='italic')
plt.savefig('figures/fig3_task_scatter.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"\nSaved figures/fig3_task_scatter.png  (font: {font})")

# Save CSV
out_df = sel_df[['dwa_id', 'description', 'cluster_name', 'x', 'y']]
out_df.columns = ['dwa_id', 'description', 'cluster_name', 'semantic_x', 'semantic_y']
out_df.to_csv('figures/fig3_selected_dwas.csv', index=False)
print(f"Saved figures/fig3_selected_dwas.csv  ({len(out_df)} DWAs)")

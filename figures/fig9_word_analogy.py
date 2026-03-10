"""
Figure 9 — Word analogy illustration (King-Queen parallelogram)
Target: Slide 3 (right panel)
Classic word embedding analogy from Mikolov et al. (2013).
This is an ILLUSTRATION of the concept, not computed from MPNet.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
from figures.style import (setup, PRIMARY, DARK, MID, GRID,
                           FONT_TITLE, FONT_TICK, add_subtitle)

font = setup()

# Illustrative positions — approximate parallelogram (not perfect)
# Based on typical Word2Vec PCA projections from literature
points = {
    'Man':    (1.5, 1.0),
    'Woman':  (4.1, 1.3),
    'King':   (1.8, 3.2),
    'Queen':  (4.3, 3.6),
}

fig, ax = plt.subplots(figsize=(4.5, 4.0))

# Grid
ax.grid(True, color=GRID, lw=0.5, zorder=0)
ax.set_xlim(0, 5.5)
ax.set_ylim(0, 4.8)
ax.tick_params(axis='both', labelsize=FONT_TICK)
ax.set_aspect('equal')

# Points
for word, (x, y) in points.items():
    ax.scatter(x, y, s=80, c=PRIMARY, edgecolors='white', linewidths=0.8, zorder=4)
    # Offset labels to avoid overlapping dots
    offset = (8, 8) if word in ('King', 'Man') else (-8, 8)
    ha = 'left' if word in ('King', 'Man') else 'right'
    ax.annotate(word, xy=(x, y), xytext=offset, textcoords='offset points',
                fontsize=FONT_TITLE, fontweight='bold', color=DARK,
                ha=ha, va='bottom', zorder=5)

# Dashed arrows showing the two relationship vectors
arrow_kw = dict(arrowstyle='->', color=MID, lw=1.2, linestyle='--',
                shrinkA=8, shrinkB=8)

# Man → Woman (gender direction)
ax.annotate('', xy=points['Woman'], xytext=points['Man'],
            arrowprops=arrow_kw, zorder=3)
mid_mw = ((points['Man'][0] + points['Woman'][0]) / 2,
          (points['Man'][1] + points['Woman'][1]) / 2)
ax.text(mid_mw[0], mid_mw[1] - 0.25, 'gender', ha='center', fontsize=FONT_TICK,
        color=MID, fontstyle='italic')

# Man → King (royalty direction)
ax.annotate('', xy=points['King'], xytext=points['Man'],
            arrowprops=arrow_kw, zorder=3)
mid_mk = ((points['Man'][0] + points['King'][0]) / 2,
          (points['Man'][1] + points['King'][1]) / 2)
ax.text(mid_mk[0] - 0.35, mid_mk[1], 'royalty', ha='center', fontsize=FONT_TICK,
        color=MID, fontstyle='italic', rotation=75)

# King → Queen (same gender direction, completing the parallelogram)
ax.annotate('', xy=points['Queen'], xytext=points['King'],
            arrowprops=dict(arrowstyle='->', color=PRIMARY, lw=1.5, linestyle='-',
                            shrinkA=8, shrinkB=8),
            zorder=3)

# Remove axis labels (grid numbers are enough)
ax.set_xlabel('')
ax.set_ylabel('')

# Attribution + formula subtitle
add_subtitle(fig,
             r'$\mathbf{king} - \mathbf{man} + \mathbf{woman} \approx \mathbf{queen}$'
             + '\n' + 'Mikolov et al. (2013) \u2014 directions encode meaning',
             y=-0.08, fontsize=FONT_TICK)

plt.tight_layout()
plt.savefig('figures/fig9_word_analogy.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"Saved figures/fig9_word_analogy.png  (font: {font})")

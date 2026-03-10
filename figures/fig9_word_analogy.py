"""
Figure 9 — Word analogy illustration (Atlanta/Denver parallelogram)
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
points = {
    'Georgia':  (1.3, 1.0),
    'Colorado': (4.0, 1.4),
    'Atlanta':  (1.7, 3.3),
    'Denver':   (4.5, 3.5),
}

fig, ax = plt.subplots(figsize=(4.5, 4.0))

# Grid
ax.grid(True, color=GRID, lw=0.5, zorder=0)
ax.set_xlim(0, 5.5)
ax.set_ylim(0, 4.8)
ax.tick_params(axis='both', labelsize=FONT_TICK)
ax.set_aspect('equal')

# Points and labels
for word, (x, y) in points.items():
    ax.scatter(x, y, s=80, c=PRIMARY, edgecolors='white', linewidths=0.8, zorder=4)
    # Top row (cities) — labels above; bottom row (states) — labels below
    if word in ('Atlanta', 'Denver'):
        offset = (0, 10)
        va = 'bottom'
    else:
        offset = (0, -10)
        va = 'top'
    ax.annotate(word, xy=(x, y), xytext=offset, textcoords='offset points',
                fontsize=FONT_TITLE, fontweight='bold', color=DARK,
                ha='center', va=va, zorder=5)

# Grey dashed path: Atlanta → Georgia → Colorado → Denver
path_arrow = dict(arrowstyle='->', color=MID, lw=1.0, linestyle='--',
                  shrinkA=4, shrinkB=4)

# Atlanta → Georgia  (the "− Georgia" step)
ax.annotate('', xy=points['Georgia'], xytext=points['Atlanta'], arrowprops=path_arrow, zorder=3)

# Georgia → Colorado  (the "+ Colorado" step)
ax.annotate('', xy=points['Colorado'], xytext=points['Georgia'], arrowprops=path_arrow, zorder=3)

# Colorado → Denver  (the "≈ Denver" arrival)
ax.annotate('', xy=points['Denver'], xytext=points['Colorado'], arrowprops=path_arrow, zorder=3)

# Blue solid arrow: Atlanta → Denver (the direct result)
ax.annotate('', xy=points['Denver'], xytext=points['Atlanta'],
            arrowprops=dict(arrowstyle='->', color=PRIMARY, lw=1.5,
                            shrinkA=4, shrinkB=4), zorder=3)

# Remove axis labels (grid numbers are enough)
ax.set_xlabel('')
ax.set_ylabel('')

# Attribution + formula subtitle
add_subtitle(fig,
             r'$\mathbf{Atlanta} - \mathbf{Georgia} + \mathbf{Colorado} \approx \mathbf{Denver}$'
             + '\n' + 'Mikolov et al. (2013) \u2014 directions encode meaning',
             y=-0.08, fontsize=FONT_TICK)

plt.tight_layout()
plt.savefig('figures/fig9_word_analogy.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"Saved figures/fig9_word_analogy.png  (font: {font})")

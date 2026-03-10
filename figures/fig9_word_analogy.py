"""
Figure 9 — Word analogy (King-Queen) parallelogram
Target: Slide 3 (right panel) — What is a Sentence Embedding?
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt

from figures.style import (setup, PRIMARY, MID, DARK, GRID,
                           add_subtitle, FONT_TITLE, FONT_TICK)

font = setup()

# Real approximate parallelogram positions
points = {
    'King':  (1.8, 3.2),
    'Queen': (4.3, 3.6),
    'Man':   (1.5, 1.0),
    'Woman': (4.1, 1.3),
}

fig, ax = plt.subplots(figsize=(4.5, 4.0))
ax.set_xlim(0, 5.5)
ax.set_ylim(0, 4.5)
ax.grid(True, color=GRID, lw=0.5)
ax.set_xticks([0, 1, 2, 3, 4, 5])
ax.set_yticks([0, 1, 2, 3, 4])
ax.tick_params(axis='both', labelsize=FONT_TICK)

# --- Dashed arrows ---
arrow_kw = dict(arrowstyle='->', color=MID, lw=1.2, linestyle='dashed')

# Man → King ("royalty")
mx, my = points['Man']
kx, ky = points['King']
ax.annotate('', xy=(kx, ky), xytext=(mx, my), arrowprops=arrow_kw)
ax.text((mx + kx) / 2 - 0.35, (my + ky) / 2, 'royalty',
        fontsize=FONT_TICK, color=MID, ha='center', va='center',
        rotation=74, fontstyle='italic')

# Man → Woman ("gender")
wx, wy = points['Woman']
ax.annotate('', xy=(wx, wy), xytext=(mx, my), arrowprops=arrow_kw)
ax.text((mx + wx) / 2, (my + wy) / 2 - 0.25, 'gender',
        fontsize=FONT_TICK, color=MID, ha='center', va='center',
        fontstyle='italic')

# King → Queen (same direction as Man→Woman, no label)
qx, qy = points['Queen']
ax.annotate('', xy=(qx, qy), xytext=(kx, ky),
            arrowprops=dict(arrowstyle='->', color=PRIMARY, lw=1.5,
                            linestyle='dashed'))

# --- Points ---
for name, (x, y) in points.items():
    ax.plot(x, y, 'o', color=PRIMARY, markersize=10, zorder=3)
    if name in ('King', 'Queen'):
        ax.text(x, y + 0.25, name, fontsize=FONT_TITLE, color=DARK,
                ha='center', va='bottom', fontweight='bold')
    else:
        ax.text(x, y - 0.25, name, fontsize=FONT_TITLE, color=DARK,
                ha='center', va='top', fontweight='bold')

# Formula subtitle
add_subtitle(fig,
             r'$\mathbf{king} - \mathbf{man} + \mathbf{woman} \approx \mathbf{queen}$',
             y=-0.06, fontsize=FONT_TITLE)

plt.savefig('figures/fig9_word_analogy.png', dpi=300,
            bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print(f"Saved figures/fig9_word_analogy.png  (font: {font})")

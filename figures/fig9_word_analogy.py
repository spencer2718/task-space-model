"""
Figure 9 — Word analogy (King-Queen) parallelogram
Target: Slide 3 (right panel) — What is a Sentence Embedding?
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt

from figures.style import setup, PRIMARY, MID, DARK, add_subtitle

font = setup()

points = {
    'King':  (2.0, 3.5),
    'Queen': (4.0, 3.5),
    'Man':   (2.0, 1.5),
    'Woman': (4.0, 1.5),
}

fig, ax = plt.subplots(figsize=(4.5, 4.0))
ax.set_xlim(0.5, 5.5)
ax.set_ylim(0.3, 4.7)
ax.axis('off')

# --- Dashed arrows ---
arrow_kw = dict(arrowstyle='->', color=MID, lw=1.2, linestyle='dashed')

# Man → King (vertical — "royalty")
ax.annotate('', xy=points['King'], xytext=points['Man'],
            arrowprops=arrow_kw)
ax.text(1.55, 2.5, 'royalty', fontsize=9, color=MID,
        ha='center', va='center', rotation=90, fontstyle='italic')

# Man → Woman (horizontal — "gender")
ax.annotate('', xy=points['Woman'], xytext=points['Man'],
            arrowprops=arrow_kw)
ax.text(3.0, 1.15, 'gender', fontsize=9, color=MID,
        ha='center', va='center', fontstyle='italic')

# King → Queen (horizontal — same "gender" vector)
ax.annotate('', xy=points['Queen'], xytext=points['King'],
            arrowprops=dict(arrowstyle='->', color=PRIMARY, lw=1.5,
                            linestyle='dashed'))
ax.text(3.0, 3.85, 'gender', fontsize=9, color=PRIMARY,
        ha='center', va='center', fontstyle='italic', fontweight='bold')

# --- Points ---
for name, (x, y) in points.items():
    ax.plot(x, y, 'o', color=PRIMARY, markersize=10, zorder=3)
    # Position labels to avoid arrow overlap
    if name in ('King', 'Queen'):
        ax.text(x, y + 0.3, name, fontsize=11, color=DARK,
                ha='center', va='bottom', fontweight='bold')
    else:
        ax.text(x, y - 0.3, name, fontsize=11, color=DARK,
                ha='center', va='top', fontweight='bold')

add_subtitle(fig, 'Directions in embedding space encode semantic relationships',
             y=-0.02)

plt.savefig('figures/fig9_word_analogy.png', dpi=300,
            bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print(f"Saved figures/fig9_word_analogy.png  (font: {font})")

"""
Figure 9 — Word analogy (King-Queen) parallelogram
Target: Slide 3 (right panel) — What is a Sentence Embedding?
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from figures.style import (setup, PRIMARY, MID, DARK, GRID,
                           add_subtitle, FONT_TITLE, FONT_TICK)
from src.task_space.data.artifacts import get_embeddings

font = setup()

# === Embed the four words live ===
words = ['king', 'queen', 'man', 'woman']
word_embs = get_embeddings(words, model='all-mpnet-base-v2')

# Project to 2D via PCA
pca = PCA(n_components=2)
coords_2d = pca.fit_transform(word_embs)

# Build points dict (capitalize for display)
points = {w.capitalize(): (coords_2d[i, 0], coords_2d[i, 1])
          for i, w in enumerate(words)}

# Print for verification
print("PCA-projected positions:")
for w, (x, y) in points.items():
    print(f'  {w:>8s}: ({x:+.3f}, {y:+.3f})')

# Check for degeneracy
all_x = [p[0] for p in points.values()]
all_y = [p[1] for p in points.values()]
x_span = max(all_x) - min(all_x)
y_span = max(all_y) - min(all_y)
aspect = min(x_span, y_span) / max(x_span, y_span) if max(x_span, y_span) > 0 else 0
print(f"  x_span={x_span:.3f}, y_span={y_span:.3f}, aspect={aspect:.3f}")
if aspect < 0.1:
    print("STOP: PCA projection is degenerate (nearly collinear). Report to Lead Researcher.")
    sys.exit(1)

# Verify the analogy
king_vec, queen_vec, man_vec, woman_vec = word_embs
result = king_vec - man_vec + woman_vec
cos_sim = np.dot(result, queen_vec) / (np.linalg.norm(result) * np.linalg.norm(queen_vec))
print(f'  cos_sim(king - man + woman, queen) = {cos_sim:.4f}')

# === Figure ===
fig, ax = plt.subplots(figsize=(4.5, 4.0))

margin = 0.5
ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
ax.grid(True, color=GRID, lw=0.5)
ax.tick_params(axis='both', labelsize=FONT_TICK)

# --- Dashed arrows ---
arrow_kw = dict(arrowstyle='->', color=MID, lw=1.2, linestyle='dashed')

# Man → King ("royalty")
mx, my = points['Man']
kx, ky = points['King']
ax.annotate('', xy=(kx, ky), xytext=(mx, my), arrowprops=arrow_kw)
# Label at midpoint, offset perpendicular to arrow (always right-side-up)
mid_rk = ((mx + kx) / 2, (my + ky) / 2)
dx_rk, dy_rk = kx - mx, ky - my
angle_rk = np.degrees(np.arctan2(dy_rk, dx_rk))
# Keep text right-side-up: if angle > 90 or < -90, flip by 180
disp_angle_rk = angle_rk + 180 if angle_rk > 90 else (angle_rk - 180 if angle_rk < -90 else angle_rk)
# Perpendicular offset (to the right of travel direction)
perp_rk = np.radians(angle_rk + 90)
off = 0.12
ax.text(mid_rk[0] + off * np.cos(perp_rk),
        mid_rk[1] + off * np.sin(perp_rk),
        'royalty', fontsize=FONT_TICK, color=MID,
        ha='center', va='center', rotation=disp_angle_rk, fontstyle='italic')

# Man → Woman ("gender")
wx, wy = points['Woman']
ax.annotate('', xy=(wx, wy), xytext=(mx, my), arrowprops=arrow_kw)
mid_gd = ((mx + wx) / 2, (my + wy) / 2)
dx_gd, dy_gd = wx - mx, wy - my
angle_gd = np.degrees(np.arctan2(dy_gd, dx_gd))
disp_angle_gd = angle_gd + 180 if angle_gd > 90 else (angle_gd - 180 if angle_gd < -90 else angle_gd)
perp_gd = np.radians(angle_gd + 90)
ax.text(mid_gd[0] + off * np.cos(perp_gd),
        mid_gd[1] + off * np.sin(perp_gd),
        'gender', fontsize=FONT_TICK, color=MID,
        ha='center', va='center', rotation=disp_angle_gd, fontstyle='italic')

# King → Queen (same direction as Man→Woman, no label)
qx, qy = points['Queen']
ax.annotate('', xy=(qx, qy), xytext=(kx, ky),
            arrowprops=dict(arrowstyle='->', color=PRIMARY, lw=1.5,
                            linestyle='dashed'))

# --- Points ---
for name, (x, y) in points.items():
    ax.plot(x, y, 'o', color=PRIMARY, markersize=10, zorder=3)
    # Labels above for top points, below for bottom
    if y > np.median(all_y):
        ax.text(x, y + 0.12, name, fontsize=FONT_TITLE, color=DARK,
                ha='center', va='bottom', fontweight='bold')
    else:
        ax.text(x, y - 0.12, name, fontsize=FONT_TITLE, color=DARK,
                ha='center', va='top', fontweight='bold')

# Formula subtitle
add_subtitle(fig,
             r'$\mathbf{king} - \mathbf{man} + \mathbf{woman} \approx \mathbf{queen}$',
             y=-0.06, fontsize=FONT_TITLE)

plt.savefig('figures/fig9_word_analogy.png', dpi=300,
            bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print(f"Saved figures/fig9_word_analogy.png  (font: {font})")

"""
Figure 1 — AI Exposure: Theoretical vs Observed
Target: Slide 2 (Motivation)
Data: Massenkoff & McCrory (2026), Figure 2.
Values estimated from published radial chart; see footnote.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
import numpy as np
from figures.style import setup, lighten, PRIMARY, SECONDARY, DARK, MID, FONT_TITLE, FONT_LABEL, FONT_TICK, FONT_NOTE

font = setup()

# SOC major groups — sorted by theoretical exposure (descending)
categories = [
    "Computer & Math",
    "Office & Admin",
    "Business & Financial",
    "Legal",
    "Management",
    "Education",
    "Healthcare",
    "Construction",
]

theoretical = [94, 90, 85, 80, 60, 50, 40, 15]
observed    = [33, 25, 20, 15,  8,  5,  4,  2]

n = len(categories)
y = np.arange(n)
bar_h = 0.35

fig, ax = plt.subplots(figsize=(5.5, 3.5))

# With invert_yaxis(), y - offset is visually ABOVE y + offset
# Theoretical (larger) on top, observed (smaller) below
bars_t = ax.barh(y - bar_h/2, theoretical, bar_h, color=lighten(SECONDARY, 0.4),
                 edgecolor='none', label='Theoretical capability', zorder=2)
bars_o = ax.barh(y + bar_h/2, observed, bar_h, color=PRIMARY,
                 edgecolor='none', label='Observed usage', zorder=2)

# Value labels
for i, (t, o) in enumerate(zip(theoretical, observed)):
    ax.text(t + 1.5, i - bar_h/2, f'{t}%', va='center', fontsize=FONT_TICK,
            color=SECONDARY, fontweight='bold')
    ax.text(o + 1.5, i + bar_h/2, f'{o}%', va='center', fontsize=FONT_TICK,
            color=PRIMARY, fontweight='bold')

ax.set_yticks(y)
ax.set_yticklabels(categories, fontsize=FONT_TICK)
ax.set_xlim(0, 110)
ax.set_xlabel('Share of job tasks (%)', fontsize=FONT_LABEL)
ax.tick_params(axis='x', labelsize=FONT_TICK)
ax.tick_params(axis='y', length=0)
ax.invert_yaxis()
ax.legend(fontsize=FONT_NOTE, loc='lower right', frameon=False)

# Footnote
fig.text(0.5, -0.06,
         'Source: Massenkoff & McCrory (2026), Fig. 2. Observed values estimated\n'
         'from published chart; occupation-level data not yet released as CSV.',
         ha='center', fontsize=FONT_NOTE - 1, color=MID, fontstyle='italic')

plt.tight_layout()
plt.savefig('figures/fig1_ai_exposure.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"Saved figures/fig1_ai_exposure.png  (font: {font})")

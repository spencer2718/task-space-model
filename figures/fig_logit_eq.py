"""
Logit equation — rendered via matplotlib mathtext
Target: Slide 6 (What Drives the Improvement?)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
from figures.style import setup, DARK

font = setup()

fig, ax = plt.subplots(figsize=(7.0, 0.5))
ax.axis('off')

ax.text(0.5, 0.5,
        r'$P(j \mid i) \;\propto\; \exp\left(-\alpha \cdot d_{\mathrm{sem}}(i,j) '
        r'\;-\; \beta \cdot d_{\mathrm{inst}}(i,j)\right)$',
        ha='center', va='center', fontsize=18, color=DARK,
        transform=ax.transAxes)

plt.savefig('figures/fig_logit_eq.png', dpi=300, bbox_inches='tight',
            facecolor='none', edgecolor='none', transparent=True)
plt.close()
print(f"Saved figures/fig_logit_eq.png  (font: {font})")

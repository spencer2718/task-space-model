"""
Figure 1 — AI task exposure by SOC major group (horizontal bar chart)
Target: Slide 2 ("Motivation")
Data: Eloundou et al. (2023) γ = E1 + E2 (share of tasks feasible with LLM + software)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import matplotlib.pyplot as plt
from figures.style import setup, PRIMARY, MID, DARK, add_subtitle

font = setup()

# === SOC labels ===
soc_names = {
    '11': 'Management',          '13': 'Business & Financial',
    '15': 'Computer & Math',     '17': 'Architecture & Eng.',
    '19': 'Life/Phys./Social Sci.', '21': 'Community & Social Svc.',
    '23': 'Legal',               '25': 'Education & Library',
    '27': 'Arts, Design & Media','29': 'Healthcare Practitioners',
    '31': 'Healthcare Support',  '33': 'Protective Service',
    '35': 'Food Prep & Serving', '37': 'Building & Grounds',
    '39': 'Personal Care & Svc.','41': 'Sales',
    '43': 'Office & Admin.',     '45': 'Farming/Fishing/Forestry',
    '47': 'Construction & Extr.','49': 'Installation & Maint.',
    '51': 'Production',          '53': 'Transportation & Moving',
}

# === Load and aggregate ===
df = pd.read_csv('data/external/eloundou/occ_level.csv')
df['soc2'] = df['O*NET-SOC Code'].str[:2]
agg = df.groupby('soc2')['dv_rating_gamma'].mean().reset_index()
agg.columns = ['soc2', 'gamma']
agg = agg[agg['soc2'].isin(soc_names)]
agg['name'] = agg['soc2'].map(soc_names)

# Sort ascending so highest is at top of horizontal bar chart
d = agg.sort_values('gamma', ascending=True).reset_index(drop=True)
n = len(d)

fig, ax = plt.subplots(figsize=(9.0, 6.5))

ax.barh(range(n), d['gamma'].values * 100, height=0.65,
        color=PRIMARY, alpha=0.85, edgecolor='none', zorder=2)

for i, row in d.iterrows():
    ax.text(-0.5, i, row['name'], ha='right', va='center',
            fontsize=11, color=DARK)
    ax.annotate(f'{row["gamma"] * 100:.0f}%',
                xy=(row['gamma'] * 100, i), xycoords='data',
                xytext=(6, 0), textcoords='offset points',
                ha='left', va='center', fontsize=10, color=MID,
                fontweight='bold')

ax.set_xlim(0, 110)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=11)
ax.set_yticks([])
ax.set_xlabel('Share of tasks theoretically feasible (γ)', fontsize=12,
              labelpad=8)
ax.tick_params(axis='y', length=0)
ax.tick_params(axis='x', length=4)

add_subtitle(fig, 'Eloundou et al. (2023) γ = E1 + E2: LLM + complementary software', y=0.005)

plt.tight_layout()
plt.savefig('figures/fig1_ai_exposure.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"Saved figures/fig1_ai_exposure.png  (font: {font})")

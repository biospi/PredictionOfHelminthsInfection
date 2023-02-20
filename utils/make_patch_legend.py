from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import datetime
import numpy as np

colors = plt.cm.twilight(np.linspace(0, 1, 12))

map_color = {1: colors[0],
             2: colors[1],
             3: colors[2],
             4: colors[3],
             5: colors[4],
             6: colors[5],
             7: colors[6],
             8: colors[7],
             9: colors[8],
             10: colors[9],
             11: colors[10],
             12: colors[11]}

legend_elements = []
for k, v in map_color.items():
    legend_elements.append(Patch(facecolor=v, edgecolor="black", alpha=0.6,
                         label=datetime.date(1900, k, 1).strftime('%B')))

# Create the figure
fig, ax = plt.subplots(figsize=(12.8, 7.2))
ax.legend(handles=legend_elements, loc='center', ncol=int(len(legend_elements)/2))

plt.tight_layout()
plt.show()
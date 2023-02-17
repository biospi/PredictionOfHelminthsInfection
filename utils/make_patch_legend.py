from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import datetime

map_color = {1: 'tab:blue',
             2: 'tab:orange',
             3: 'tab:red',
             4: 'tab:green',
             5: 'white',
             6: 'tab:brown',
             7: 'tab:pink',
             8: 'tab:gray',
             9: 'tab:olive',
             10: 'tab:cyan',
             11: 'black',
             12: 'tab:purple'}

legend_elements = []
for k, v in map_color.items():
    legend_elements.append(Patch(facecolor=v, edgecolor="black", alpha=0.5,
                         label=datetime.date(1900, k, 1).strftime('%B')))

# Create the figure
fig, ax = plt.subplots(figsize=(12.8, 7.2))
ax.legend(handles=legend_elements, loc='center', ncol=int(len(legend_elements)/2))

plt.tight_layout()
plt.show()
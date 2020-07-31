from datetime import datetime

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize, minmax_scale
import pandas as pd
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter


def plot_stacked_bar(data, series_labels, category_labels=None,
                     show_values=False, value_format="{}", y_label=None, x_label=None,
                     colors=None, grid=False, reverse=False):
    """Plots a stacked bar chart with the data and labels provided.

    Keyword arguments:
    data            -- 2-dimensional numpy array or nested list
                       containing data for each series in rows
    series_labels   -- list of series labels (these appear in
                       the legend)
    category_labels -- list of category labels (these appear
                       on the x-axis)
    show_values     -- If True then numeric value labels will
                       be shown on each bar
    value_format    -- Format string for numeric value labels
                       (default is "{}")
    y_label         -- Label for y-axis (str)
    colors          -- List of color labels
    grid            -- If True display grid
    reverse         -- If True reverse the order that the
                       series are displayed (left-to-right
                       or right-to-left)
    """

    ny = len(data[0])
    ind = list(range(ny))

    axes = []
    cum_size = np.zeros(ny)

    data = np.array(data)

    if reverse:
        data = np.flip(data, axis=1)
        category_labels = reversed(category_labels)

    for i, row_data in enumerate(data):
        color = colors[i] if colors is not None else None
        axes.append(plt.bar(ind, row_data, bottom=cum_size,
                            label=series_labels[i], color=color))
        cum_size += row_data

    if category_labels:
        plt.xticks(ind, category_labels)

    if y_label:
        plt.ylabel(y_label)

    if x_label:
        plt.xlabel(x_label)

    plt.legend()

    if grid:
        plt.grid()

    if show_values:
        for axis in axes:
            for bar in axis:
                w, h = bar.get_width(), bar.get_height()
                plt.text(bar.get_x() + w/2, bar.get_y() + h/2,
                         value_format.format(h), ha="center",
                         va="center")
    plt.gcf().autofmt_xdate()


def create_histogram(input, farm_id):
    print("create histogram...")
    new_keys = ["("+str(list(input.keys())[n])+")  "+str(x) for n, x in enumerate(range(len(input.keys())))]
    catalog = {}
    for animal_id, value in input.items():
        valid = value['valid']
        for v in valid:
            if animal_id not in catalog:
                catalog[animal_id] = {"valid_cpt": 0, "not_valid_cpt": 0, "sample_count": 0}
            catalog[animal_id]['sample_count'] = catalog[animal_id]['sample_count'] + 1
            if v:
                catalog[animal_id]['valid_cpt'] = catalog[animal_id]['valid_cpt'] + 1
            else:
                catalog[animal_id]['not_valid_cpt'] = catalog[animal_id]['not_valid_cpt'] + 1

    plt.figure(figsize=(12, 8))
    catalog_sorted = dict(sorted(catalog.items(), key=lambda x: x[1]['sample_count'], reverse=True))

    data = [
        [d['valid_cpt'] for d in catalog_sorted.values()],
        [d['not_valid_cpt'] for d in catalog_sorted.values()]
    ]

    category_labels = new_keys

    plot_stacked_bar(
        data,
        ['Valid', 'Not Valid'],
        category_labels=category_labels,
        show_values=True,
        value_format="{:.0f}",
        colors=['tab:blue', 'tab:orange'],
        y_label="Number of samples",
        x_label="Animal ID"
    )

    print('saving fig...')

    plt.savefig('%s.png' % farm_id)
    plt.show()


def create_dataset_map(data, farm_id, fontsize=20):
    print("create dataset map...")

    new_keys = ["("+str(list(data.keys())[n])+")  "+str(x) for n, x in enumerate(range(len(data.keys())))]
    old_keys = data.copy().keys()
    for new_key, old_key in zip(new_keys, old_keys): #need to clean up keys for later use below
        data[new_key] = data.pop(old_key)

    fig, ax = plt.subplots(figsize=(40, 5))
    a_data, t_data = [], []
    animals_id = data.keys()
    longes_xaxis = []
    for animal_id, value in data.items():
        activity = value['activity']
        time_axis = value['date']
        famacha = value['famacha']
        valid = value['valid']

        for i, _ in enumerate(activity):
            if not valid[i]:
                continue

            color = 'lime'
            if famacha[i] == 2:
                color = 'red'
            if famacha[i] > 2:
                color = 'orange'

            ax.add_patch(Rectangle((i*len(activity[0]), int(animal_id.split(')')[-1])), len(activity[0]),
                                   1, fill=False, edgecolor=color, facecolor=None, lw=1, alpha=None))

        a_data.append(sum(activity, []))
        xaxis = sum(time_axis, [])
        if(len(xaxis) > len(longes_xaxis)):
            longes_xaxis = xaxis

    a_data = pd.DataFrame(a_data).values
    a_f = a_data[~np.isnan(a_data)]
    print("acrivity data range", a_f.min(), a_f.max())
    MAX_VALUE = np.median(a_f)
    a_data[a_data > 255] = MAX_VALUE

    ax.set_facecolor('xkcd:gray')
    ax.set_title(farm_id)
    ax.set_yticks(np.arange(len(animals_id)))
    ax.set_yticklabels(animals_id)

    date_tick_label = [i.strftime("%y/%m/%d %H:%M") for i in longes_xaxis]
    x = list(range(len(date_tick_label)))
    p = int(len(x) / 100)
    if p <= 0:
        p = 2
    ax.set_xticks(x[::p])
    ax.set_xticklabels(date_tick_label[::p])
    ax.pcolormesh(a_data)

    fig.autofmt_xdate()
    fig.show()
    print('saving fig...')
    fig.savefig('%s.png' % farm_id)
    print("saved!")


def create_herd_map(farm_id, meta_data, activity_data, animals_id, time_range, fontsize=20):
    print("create_herd_map...")
    fig, ax = plt.subplots(figsize=(200, 200))
    MAX = 200
    activity_data[activity_data > MAX] = MAX
    im = ax.pcolormesh(activity_data)

    # We want to show all ticks...
    if time_range is not None:
        ax.set_xticks(np.arange(len(time_range)))
    ax.set_yticks(np.arange(len(animals_id)))
    # ... and label them with the respective list entries
    if time_range is not None:
        ax.set_xticklabels(time_range)
    ax.set_yticklabels(animals_id)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    if time_range is not None:
        if meta_data is not None:
            for i in range(len(animals_id)):
                for j in range(len(time_range)):
                    color = "w"
                    if meta_data[i, j] == 0:
                        continue
                    if meta_data[i, j] >= 2:
                        color = 'r'
                    if meta_data[i, j] == 1:
                        color = 'g'
                    text = ax.text(j, i, str(meta_data[i, j]),
                                   ha="center", va="center", color=color, fontsize=fontsize)

    # ax.set_title(farm_id)
    # fig.tight_layout()
    plt.show()
    print('saving fig...')
    fig.savefig('%s.png' % farm_id)
    print("saved!")


if __name__ == "__main__":
    meta_data = np.array([[-1, -1, -1, 1, 0, 0, 2],
                          [0, 3, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0]])

    activity_data = np.array(
        [
            [0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
            [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
            [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0]
        ]
    )
    animals_id = ["0", "1", "2"]
    time_range = [datetime(2013, 2, 1, 0, 0), datetime(2013, 2, 2, 0, 0), datetime(2013, 2, 3, 0, 0),
                  datetime(2013, 2, 4, 0, 0), datetime(2013, 2, 5, 0, 0), datetime(2013, 2, 6, 0, 0),
                  datetime(2013, 2, 7, 0, 0)]

    time_range_ = [datetime(2013, 2, 8, 0, 0), datetime(2013, 2, 9, 0, 0), datetime(2013, 2, 10, 0, 0),
                  datetime(2013, 2, 11, 0, 0), datetime(2013, 2, 12, 0, 0), datetime(2013, 2, 13, 0, 0),
                  datetime(2013, 2, 14, 0, 0)]

    time_range__ = [datetime(2013, 2, 15, 0, 0), datetime(2013, 2, 16, 0, 0), datetime(2013, 2, 17, 0, 0),
                   datetime(2013, 2, 18, 0, 0), datetime(2013, 2, 19, 0, 0), datetime(2013, 2, 20, 0, 0),
                   datetime(2013, 2, 21, 0, 0)]
    farm_id = "Delmas"

    data = {"0": {"activity": [[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0], [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0]],
                  "date": [time_range, time_range_], "famacha": [1, 2], "valid": [False, True]},
            "1": {"activity": [[2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0]], "date": [time_range], "famacha": [1], "valid": [True]},
            "2": {"activity": [[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0], [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0]],
                  "date": [time_range, time_range_], "famacha": [2, 3], "valid": [True, True]},
            "3": {"activity": [[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0], [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0], [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0]],
                  "date": [time_range, time_range_, time_range__], "famacha": [2, 3, 1], "valid": [True, False, True]}
            }

    # create_herd_map(farm_id, meta_data, activity_data, animals_id, time_range, fontsize=50)

    create_histogram(data, "farm_id")
    create_dataset_map(data, "farm_id")

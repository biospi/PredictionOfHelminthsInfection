
from datetime import datetime, timedelta, date

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize, minmax_scale
import pandas as pd
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import matplotlib.patches as mpatches
from sys import exit
from itertools import groupby
from natsort import natsorted


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

    total_valid = sum([d['valid_cpt'] for d in catalog_sorted.values()])
    total_not_valid = sum([d['not_valid_cpt'] for d in catalog_sorted.values()])

    category_labels = new_keys

    plot_stacked_bar(
        data,
        ['Valid (%d)' % total_valid, 'Not Valid (%d)' % total_not_valid],
        category_labels=category_labels,
        show_values=True,
        value_format="{:.0f}",
        colors=['tab:blue', 'tab:orange'],
        y_label="Number of samples",
        x_label="Animal ID"
    )

    print('saving fig...')
    path = '%s.png' % farm_id
    print(path)
    plt.savefig(path)
    plt.show()


def divide_chunks(l, n, reduced=True, dates=False):
    # looping till length l
    for i in range(0, len(l), n):
        if dates:
            data = l[i:i + n].values
            yield (data[0], data[-1])
            continue
        if reduced:
            yield l[i:i + n].values[0]
            continue

        yield l[i:i + n].values


def mark_to_del(row):
    a_keys = []
    for key in row.keys():
        if str(key).isdigit():
            a_keys.append(key)
    values = np.array([row[x] for x in a_keys])
    non_nan = values.size - np.count_nonzero(np.isnan(values))
    if non_nan == 0:
        return 1
    return 0

cpt_del = 0
def mark_to_del2(row):
    global cpt_del
    to_del = row['del']
    index = row.name
    if to_del == 0:
        cpt_del = 0
    else:
        cpt_del += 1
    return cpt_del


def label_color(row, j):
    try:
        if row["valid_%d" % j] == 0:
            return np.nan
        if row["%d" % j] == np.nan:
            return np.nan
        if row["famacha_prev_%d" % j] == 1 and row["famacha_%d" % j] >= 2:
            return 1
        if row["famacha_prev_%d" % j] >= 2 and row["famacha_%d" % j] == 1:
            return 2
        if row["famacha_prev_%d" % j] == 2 and row["famacha_%d" % j] == 2:
            return 3
        if row["famacha_prev_%d" % j] == 1 and row["famacha_%d" % j] == 1:
            return 4
        if row["famacha_prev_%d" % j] > 2 and row["famacha_%d" % j] > 2:
            return 5
    except TypeError as e:
        print(e)
        return np.nan
    return np.nan


def pad_for_visualization(activity_data, dil=10):
    for array in activity_data:
        pad = True
        cpt_pad = 0
        for k, item in enumerate(array):
            if np.isnan(item):
                pad = True

            if pad:
                array[k] = np.nan
                cpt_pad += 1

            if cpt_pad > dil:
                cpt_pad = 0
                pad = False
    return activity_data


def create_dataset_map(data, farm_id, fontsize=20, chunck_size=7):
    print("create dataset map...")

    new_keys = [str(list(data.keys())[n])[-4:]+" "+str(x) for n, x in enumerate(range(len(data.keys())))]
    #new_keys = [str(x) for n, x in enumerate(range(len(data.keys())))]
    old_keys = data.copy().keys()
    for new_key, old_key in zip(new_keys, old_keys): #need to clean up keys for later use below
        data[new_key] = data.pop(old_key)

    fig, ax = plt.subplots(figsize=(10, 5))
    # a_data, t_data = [], []
    animals_id = data.keys()
    #longes_xaxis = []
    cpt_12 = 0
    cpt_21 = 0
    cpt_22 = 0
    cpt_11 = 0
    cpt_nan = 0
    df1 = None
    animal_ids = []
    for j, (animal_id, value) in enumerate(data.items()):
        # if animal_id not in ['26']:
        #     continue
        print(j, len(data.items()))
        # if j > 10:
        #     break
        animal_ids.append(animal_id)
        activity = value['activity']
        time_axis = value['date']
        famacha = value['famacha']
        famacha_prev = value['famacha_previous']
        valid = value['valid']
        # a_data.append(sum(activity, []))
        xaxis = sum(time_axis, [])
        #t_data.append(xaxis)
        famacha_list = []
        famacha_list_prev = []
        valid_list = []
        # animal_id_list = []

        activity_array = np.array(sum(activity, []))
        # non_nan = activity_array.size - np.count_nonzero(np.isnan(activity_array))
        # if non_nan == 0:
        #     continue

        for k, a in enumerate(activity):
            famacha_list += [famacha[k]] * len(a)
            famacha_list_prev += [famacha_prev[k]] * len(a)
            valid_list += [valid[k]] * len(a)
            # animal_id_list += [animal_id] * len(a)

        if df1 is None:
            df1 = pd.DataFrame()
            df1['%d' % j] = sum(activity, [])

            df1['date'] = xaxis
            df1.set_index('date', inplace=True)

            df1["famacha_%d" % j] = np.array(famacha_list)
            # df1["famacha_%d" % j] = df1["famacha_%d" % j].astype(np.uint16)

            df1["famacha_prev_%d" % j] = np.array(famacha_list_prev)
            # df1["famacha_prev_%d" % j] = df1["famacha_prev_%d" % j].astype(np.uint16)

            df1["valid_%d" % j] = np.array(valid_list)
            # df1["animal_id_%d" % j] = np.array(animal_id_list)
            # df1 = df1.fillna(-1)
            # df1["%d" % j] = df1["%d" % j].astype(np.int16)

            # df1["famacha_%d" % j] = df1["famacha_%d" % j].astype(np.int16)
            # df1["famacha_prev_%d" % j] = df1["famacha_prev_%d" % j].astype(np.int16)

            df1["c%d" % j] = df1.apply(lambda row: label_color(row, j), axis=1)
            del df1["famacha_%d" % j]
            del df1["famacha_prev_%d" % j]
            del df1["valid_%d" % j]
            print(df1.shape)
        else:
            df = pd.DataFrame()
            df['%d' % j] = sum(activity, [])
            df['date'] = xaxis
            df.set_index('date', inplace=True)
            df["famacha_%d" % j] = np.array(famacha_list)
            # df["famacha_%d" % j] = df["famacha_%d" % j].astype(np.uint16)

            df["famacha_prev_%d" % j] = np.array(famacha_list_prev)
            # df["famacha_prev_%d" % j] = df["famacha_prev_%d" % j].astype(np.uint16)
            df["valid_%d" % j] = np.array(valid_list)
            df["c%d" % j] = df.apply(lambda row: label_color(row, j), axis=1)

            # df["animal_id_%d" % j] = np.array(animal_id_list)

            del df["famacha_%d" % j]
            del df["famacha_prev_%d" % j]
            del df["valid_%d" % j]

            df1 = df1.join(df, how='outer', rsuffix='%d' % j)

            df1["del"] = df1.apply(lambda row: mark_to_del(row), axis=1)
            #df1 = df1[(df1["del"] == 0)]
            df1["del2"] = df1.apply(lambda row: mark_to_del2(row), axis=1)
            df1 = df1[(df1["del2"] <= 50000)]
            del df1["del2"]
            del df1["del"]


            # df1 = df1.fillna(-1)
            # df1["%d" % j] = df1["%d" % j].astype(np.int16)
            # df1["famacha_%d" % j] = df1["famacha_%d" % j].astype(np.int16)
            # df1["famacha_prev_%d" % j] = df1["famacha_prev_%d" % j].astype(np.int16)
            # df1 = df1.astype(np.int16)
            print(df1.shape)

        # if(len(xaxis) > len(longes_xaxis)):
        #     longes_xaxis = xaxis
    # cols = []
    # cpt = 1
    # for i, x in enumerate(df1.columns):
    #     if str(x).split('_')[0] == '0' or str(x) == '0':
    #         cols.append(str(cpt))
    #         cpt += 1
    #         continue
    #     cols.append(x)
    # df1.columns = cols
    df1 = df1.reindex(sorted(df1.columns), axis=1)
    # df1 = df1.astype(np.int64)
    #df1 = df1.fillna(method='ffill')

    # a_data = pd.DataFrame(a_data).values
    # t_data = pd.DataFrame(t_data).values

    # a_f = a_data[~np.isnan(a_data)]
    # print("acrivity data range", a_f.min(), a_f.max())
    # MAX_VALUE = np.median(a_f) * 10
    # df1[df1 > MAX_VALUE] = MAX_VALUE
    # MAX_VALUE = 100
    activity_cols = []
    print("df ready!")
    print("adding famacha layer...")

    min_date = df1.index.values[0]
    max_date = df1.index.values[-1]
    rectangles = []
    for n, col in enumerate(df1.columns):
        if str(col).isdigit():

            # animal_id = df1["animal_id_%d" % n].drop_duplicates()
            # animal_id = list(animal_id.values[~pd.isnull(animal_id.values)])[0]
            # del df1["animal_id_%d" % n]
            # time_axis = df1.index.to_list()
            famacha = df1["c%d" % int(col)].replace(np.nan, -1)
            del df1["c%d" % int(col)]

            famacha = list(divide_chunks(famacha, chunck_size))
            dates = pd.Series(df1.index)

            dates = list(divide_chunks(dates, chunck_size, dates=True))

            # famacha_prev = df1["famacha_prev_%d" % n].replace(np.nan, -1)
            # del df1["famacha_prev_%d" % n]
            # famacha_prev = list(divide_chunks(famacha_prev, chunck_size))

            # valid = df1["valid_%d" % n].replace(np.nan, False)
            # del df1["valid_%d" % n]
            # valid = list(divide_chunks(valid, chunck_size))

            activity = df1[col]
            activity_cols.append(col)
            activity = list(divide_chunks(activity, chunck_size, reduced=False))

            for i, _ in enumerate(activity):

                color = 'lime'
                if famacha[i] == 1:
                    color = 'red'
                    cpt_12 = cpt_12 + 1
                if famacha[i] == 2:
                    color = 'orange'
                    cpt_21 = cpt_21 + 1
                if famacha[i] == 3 or famacha[i] == 5:
                    color = 'cyan'
                    cpt_22 = cpt_22 + 1
                if famacha[i] == 4:
                    color = 'lime'
                    cpt_11 = cpt_11 + 1

                startTime = dates[i][0]
                endTime = startTime + np.timedelta64(7, 'D')

                start = mdates.date2num(startTime)
                end = mdates.date2num(endTime)
                width = end - start

                top_col = np.nanmax(np.array([int(x) if x.isdigit() else np.nan for x in list(df1.columns.values)]))
                y_pos = int(col)

                lw = 1
                if famacha[i] < 0:
                    lw = 1
                    color = 'grey'
                    cpt_nan = cpt_nan + 1
                    # width = 0
                    # lw = 0
                rec = Rectangle((start, y_pos), width, 1, fill=False, edgecolor=color, facecolor=None, lw=lw, alpha=None)
                if famacha[i] < 0:
                    ax.add_patch(rec)
                else:
                    rectangles.append(rec)
    print("ready to plot...")
    for r in rectangles:
        ax.add_patch(r)

    df1 = df1[activity_cols]
    df1 = df1.reindex(natsorted(df1.columns), axis=1)
    max_v = np.nanmax(df1.max(axis=1, skipna=True).values)
    min_v = np.nanmin(df1.min(axis=1, skipna=True).values)
    med_v = np.nanmedian(df1.values)
    print(max_v, min_v, med_v)

    activity_data = df1.values.T

    # activity_data = pad_for_visualization(activity_data, dil=1500)

    dates = [datetime.date(x) for x in df1.index]
    x = np.array([mdates.date2num(x) for x in dates])
    y = np.arange(activity_data.shape[0]+1)
    del df1
    # MAX_VALUE = 300
    #activity_data[activity_data > med_v] = int(med_v)
    # activity_data[activity_data < 0] = -1

    ax.set_facecolor('xkcd:gray')

    ax.set_yticks(np.arange(len(animals_id)))
    ax.set_yticklabels(animals_id)

    # assign date locator / formatter to the x-axis to get proper labels
    locator = mdates.AutoDateLocator(minticks=3)
    formatter = mdates.AutoDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%B-%d-%y"))

    # date_tick_label = [i.strftime("%B %y") for i in dates]
    # x_ = list(range(len(date_tick_label)))
    # p = int(len(x_) / 10)
    # if p <= 0:
    #     p = 2
    # ax.set_xticks(x_[::p])
    # ax.set_xticklabels(date_tick_label)

    ax.set_facecolor('xkcd:cloudy blue')

    activity_data[activity_data >= 0] = 1
    activity_data[np.isnan(activity_data)] = 0

    ax.set_yticks([float(n) + 0.5 for n in ax.get_yticks()])

    x = sorted(x, key=float)
    ax.pcolormesh(x, y, activity_data, cmap='Greys')

    title = "min=%d max=%d median=%d" % (min_v, max_v, med_v)
    ax.set_title(title)

    plt.xlim([min_date, max_date])
    fig.autofmt_xdate()

    patch1 = mpatches.Patch(color='red', label='1 -> 2 (%d)' % cpt_12)
    patch2 = mpatches.Patch(color='orange', label='2 -> 1 (%d)' % cpt_21)
    patch3 = mpatches.Patch(color='cyan', label='2 -> 2 (%d)' % cpt_22)
    patch4 = mpatches.Patch(color='lime', label='1 -> 1 (%d)' % cpt_11)
    patch5 = mpatches.Patch(color='grey', label='nan (%d)' % cpt_nan)
    plt.legend(handles=[patch1, patch2, patch3, patch4, patch5])

    fig.show()
    print('saving fig...')
    fig.savefig('%s_.png' % farm_id)
    print("saved!")
    del data


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


def test_pcolormesh_datetime_axis():
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, top=0.98, bottom=.15)
    base = datetime(2013, 1, 1)
    x = np.array([base + timedelta(days=d) for d in range(21)])
    y = np.arange(21)
    z1, z2 = np.meshgrid(np.arange(20), np.arange(20))
    z = z1 * z2
    plt.subplot(221)
    plt.pcolormesh(x[:-1], y[:-1], z)
    plt.subplot(222)
    plt.pcolormesh(x, y, z)
    x = np.repeat(x[np.newaxis], 21, axis=0)
    y = np.repeat(y[:, np.newaxis], 21, axis=1)
    plt.subplot(223)
    plt.pcolormesh(x[:-1, :-1], y[:-1, :-1], z)
    plt.subplot(224)
    plt.pcolormesh(x, y, z)
    for ax in fig.get_axes():
        for label in ax.get_xticklabels():
            label.set_ha('right')
            label.set_rotation(30)
    plt.show()


def dummy_date_range(sdate, edate):
    date_range = []
    for date in pd.date_range(start=sdate, end=edate):
        date_range.append(date.to_pydatetime())
    return date_range


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
                  "date": [time_range, time_range_], "famacha": [1, 2], "famacha_previous": [2, 1], "valid": [False, True]},
            "1": {"activity": [[2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0]], "date": [time_range__], "famacha": [1], "famacha_previous": [1], "valid": [True]},
            "2": {"activity": [[0.9, 2.4, 2.5, 3.9, 0.0, 4.2, 0.0], [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0]],
                  "date": [time_range, time_range_], "famacha": [2, 3], "famacha_previous": [2, 3], "valid": [True, True]},
            "3": {"activity": [[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                               [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]],
                  "date": [time_range, time_range_, time_range__], "famacha": [1, 1, 1], "famacha_previous": [1, 1, 1],
                  "valid": [False, False, False]},
            "4": {"activity": [[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0], [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0], [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 5.2]],
                  "date": [time_range, time_range_, time_range__], "famacha": [1, 1, 1] , "famacha_previous": [2, 2, 1], "valid": [True, False, True]}
            }

    date_range1 = dummy_date_range(date(2000, 1, 1), date(2100, 1, 1))
    date_range2 = dummy_date_range(date(2100, 1, 2), date(2200, 1, 3))
    s = len(date_range1)
    null_array = [np.nan]*s
    non_null_array = [1]*s
    null_array2 = [1]*s
    for i in range(int(s/2), int(s/2)+1):
        null_array2[i] = np.nan

    test = np.array(null_array2)

    data = {"0": {"activity": [non_null_array, non_null_array], "date": [date_range1, date_range2], "famacha": [2, 2], "famacha_previous": [1, 1], "valid": [True, True]},
            "1": {"activity": [null_array2, null_array2], "date": [date_range1, date_range2], "famacha": [2, 2], "famacha_previous": [1, 1], "valid": [False, False]}
            }

    # create_herd_map(farm_id, meta_data, activity_data, animals_id, time_range, fontsize=50)

    # create_histogram(data, "farm_id")
    create_dataset_map(data.copy(), "farm_id", chunck_size=s)
# 80302
# 36524
import datetime
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal


def resample_s(df_activity_window, sampling):
    resampled = []
    for i in range(df_activity_window.shape[0]):
        row = df_activity_window.iloc[i, :]
        row = signal.resample(row, sampling)
        resampled.append(row)
    df_ = pd.DataFrame(resampled)
    df_.columns = list(range(df_.shape[1]))
    return df_


def resample(df_activity_window, sampling):
    resampled = []
    for i in range(df_activity_window.shape[0]):
        row = df_activity_window.iloc[i, :]
        mins = [datetime.datetime.today() + datetime.timedelta(minutes=1 * x) for x in range(0, len(row))]
        row.index = mins
        df_activity_window_ = row.resample(sampling).sum()
        resampled.append(df_activity_window_)
    df_ = pd.DataFrame(resampled)
    df_.columns = list(range(df_.shape[1]))
    return df_


def load_activity_data(
    out_dir,
    meta_columns,
    filepath,
    n_activity_days,
    class_healthy,
    class_unhealthy,
    imputed_days=7,
    preprocessing_steps=None,
    farm=None,
    meta_cols_str=None,
    sampling='T',
    individual_to_ignore=[],
    individual_to_keep=[],
    plot_s_distribution=True,
    resolution = None,
    sample_date_filter = None
):
    print(f"load activity from datasets...{filepath}")
    data_frame = pd.read_csv(filepath, sep=",", header=None, low_memory=False)
    # #todo remove
    #data_frame = data_frame.iloc[0:3577, :]
    if (
        "delmas" in str(out_dir).lower()
        or "cedara" in str(out_dir).lower()
        or "cross_farm" in str(out_dir).lower()
        or "temporal" in str(out_dir).lower()
        or "cwt_explain" in str(out_dir).lower()
        or "dwt_explain" in str(out_dir).lower()
        or "test_distance_validation" in str(out_dir).lower()
    ):
        if "health" not in data_frame.columns:
            print("missing health column in dataset!")
            data_frame["health"] = 0
        if "target" not in data_frame.columns:
            print("missing target column in dataset!")
            data_frame["target"] = 0
    # # todo remove

    data_frame = data_frame.astype(
        dtype=float, errors="ignore"
    )  # cast numeric values as float
    data_point_count = data_frame.shape[1]
    hearder = [str(n) for n in range(0, data_point_count)]

    for i, m in enumerate(meta_columns[::-1]):
        hearder[-i - 1] = m
    data_frame.columns = hearder
    # data_frame['label'] = data_frame['label'].astype(int).astype(str)
    if "name" in data_frame.columns:
        data_frame["name"] = data_frame["name"].astype(str).str.replace("'", "")
    data_frame["date"] = data_frame["date"].astype(str).str.replace("'", "")
    # cast transponder ids to string instead of float
    data_frame["id"] = data_frame["id"].astype(str).str.split(".", expand=True, n=0)[0]

    if sample_date_filter is not None:
        data_frame["datetime"] = pd.to_datetime(data_frame['date'])
        data_frame = data_frame[data_frame["datetime"] > pd.Timestamp(sample_date_filter)]
        data_frame = data_frame.drop('datetime', 1)

    if len(individual_to_ignore) > 0:
        data_frame = data_frame.loc[~data_frame['name'].isin(individual_to_ignore)]#todo fix

    if len(individual_to_keep) > 0:
        data_frame = data_frame.loc[data_frame['name'].isin(individual_to_keep)]

    if n_activity_days > 0:
        df_activity_window = data_frame.iloc[:, data_frame.columns.str.isnumeric()]
        #assert sampling == "T", "activity sampling must be 1 min per bin!"
        end = int(df_activity_window.shape[1])
        start = int(end - n_activity_days * 1440)
        df_activity_window = df_activity_window.iloc[:, start:end]
        #df_activity_window = df_activity_window.iloc[:, ::-1]
        if sampling != "T":
            df_activity_window = resample(df_activity_window, sampling)

        # if resolution is not None and resolution > 0:
        #     df_activity_window = resample_s(df_activity_window, int(np.ceil(df_activity_window.shape[1] / resolution)))

        df_meta = data_frame[meta_columns]
        data_frame = pd.concat([df_activity_window, df_meta], axis=1)
        #seasons = seasons[np.repeat(seasons.columns.values, df_activity_window.shape[1])]

    if imputed_days > 0:
        data_frame = data_frame[~np.isnan(data_frame["imputed_days"])]
        data_frame = data_frame[data_frame["imputed_days"] <= imputed_days]

    data_frame = data_frame[data_frame.nunique(1) > 10]
    data_frame = data_frame.dropna(
        subset=data_frame.columns[: -len(meta_columns)], how="all"
    )
    #data_frame = data_frame.fillna(999999)
    #if imputed_days > 0:
    if len(preprocessing_steps) > 0:
        if "ZEROPAD" in preprocessing_steps[0]:
            data_frame = data_frame.fillna(0)

        if "LINEAR" in preprocessing_steps[0]:
            data_frame.iloc[:, : -len(meta_columns)] = data_frame.iloc[
                :, : -len(meta_columns)
            ].astype(float).interpolate(axis=1, limit_direction="both")

    data_frame = data_frame.dropna()

    data_frame['id'] = data_frame['id'].astype(float)
    #data_frame = data_frame[data_frame['id'] != 40]

    # clip negative values
    data_frame[data_frame.columns.values[: -len(meta_columns)]] = data_frame[
        data_frame.columns.values[: -len(meta_columns)]
    ].astype(float).clip(lower=0)

    data_frame["target"] = data_frame["target"].astype(int)
    data_frame["imputed_days"] = data_frame["imputed_days"].astype(int)
    data_frame["label"] = data_frame["label"].astype(str)
    new_label = []
    # data_frame_health = data_frame.copy()
    for v in data_frame["label"].values:
        if v in class_healthy:
            new_label.append(0)
            continue
        if v in class_unhealthy:
            new_label.append(1)
            continue
        new_label.append(-1)

    data_frame["health"] = new_label

    try:
        seasons = pd.DataFrame(
            (pd.to_datetime(data_frame["date"], format="%d/%m/%Y").dt.month % 12 // 3 + 1).values,
            columns=[data_point_count - len(meta_columns)],
            index=data_frame.index
        )
        seasons = seasons.loc[data_frame.index]
    except Exception as e: #todo check if date available
        print(e)

    # Hot Encode of FAmacha targets and assign integer target to each famacha label
    data_frame_labeled = pd.get_dummies(data_frame, columns=["label"])
    flabels = [x for x in data_frame_labeled.columns if "label" in str(x)]

    for i, flabel in enumerate(flabels):
        data_frame_labeled[flabel] = data_frame_labeled[flabel] * (i + 1)
        data_frame["target"] = data_frame["target"] + data_frame_labeled[flabel]

    # store all samples for later testing after binary fitting
    labels = data_frame["label"].drop_duplicates().values
    samples = {}

    for label in labels:
        df = data_frame[data_frame["label"] == label]
        samples[label] = df

    if plot_s_distribution:
        plot_samples_distribution(out_dir, samples, f"distrib_all_samples_{farm}.png")

    class_count = {}
    label_series = dict(data_frame[["target", "label"]].drop_duplicates().values)
    label_series_inverse = dict((v, k) for k, v in label_series.items())
    print(label_series_inverse)
    print(label_series)
    for k in label_series.keys():
        class_count[str(label_series[k]) + "_" + str(k)] = data_frame[
            data_frame["target"] == k
        ].shape[0]
    print(class_count)
    # drop label column stored previously, just keep target for ml
    meta_data = data_frame[meta_columns].values
    meta_data_short = []
    for j, m in enumerate(meta_data):
        str_f = str(j) + " "
        for i, elem in enumerate(m):
            elem = str(elem)
            m_col = meta_columns[i]
            if m_col not in meta_cols_str:
                continue
            if m_col == "id":
                elem = elem[-3:]  # only keeps the last 4 char of long id numbers
            str_f += elem + " "
        meta_data_short.append(str_f)
    meta_data_short = np.array(meta_data_short)
    print(data_frame)

    # values = data_frame.iloc[:,:-len(meta_columns)].values
    # groups = []
    # for i in range(0, values.shape[1], 1440):
    #     start = i
    #     end = start+1440
    #     print(start, end)
    #     d_ = values[:, start: end]
    #     print(d_)
    #     d_[d_==999999] = np.nan
    #     # groups.append(d_)
    #
    #     hour_nan_rate = []
    #     for j in range(0, d_.shape[1], 60):
    #         start = j
    #         end = start + 60
    #         df = np.isnan(d_[:, start:end])
    #         nan_rate = np.sum(df==True)/df.size
    #         hour_nan_rate.append(nan_rate)
    #     groups.append(np.array(hour_nan_rate))
    #     sample_nan_info = np.vstack(groups)
    #
    # fig, ax = plt.subplots()
    # im = ax.imshow(sample_nan_info, interpolation='nearest')
    # # im = ax.imshow(scores, interpolation='nearest',
    # #            norm=MidpointNormalize(vmin=-.2, midpoint=0.5))
    # ax.set_xlabel('Time (Hour of day)')
    # ax.set_ylabel('Day')
    # fig.colorbar(im)
    # # ax.set_xticks(np.arange(len(df["gamma"].unique())),
    # #            [np.format_float_scientific(i, 1) for i in df["gamma"].unique()], rotation=45)
    # # ax.set_yticks(np.arange(len(df["C"].unique()))[::-1],
    # #            [np.format_float_scientific(i, ) for i in df["C"].unique()])
    # ax.set_title(f'Rate Of Missingness (ratio of NaN)\n Per Hour Across Activity Samples')
    # fig.tight_layout()
    # fig.show()
    # out_dir.mkdir(parents=True, exist_ok=True)
    # filename = f"heatmap_sample_nan.png"
    # filepath = out_dir / filename
    # print(filepath)
    # fig.savefig(filepath)

    return (
        data_frame,
        meta_data,
        meta_data_short,
        class_healthy,
        class_unhealthy,
        label_series,
        samples,
        seasons
    )


def parse_param_from_filename(file):
    split = file.split("/")[-1].split(".")[0].split("_")
    # activity_delmas_70101200027_dbft_1_1min
    sampling = ""
    days = 0
    farm_id = "farm_id"
    option = ""
    for s in split:
        if "day" in s:
            days = int(s[0])
            break

    return days, farm_id, option, sampling


def plot_samples_distribution(out_dir, samples_, filename, grid_enabled=True):
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_data = samples_.copy()

    print(sample_data)
    # bar plot
    d = []
    for key, value in sample_data.items():
        value["id"] = [key + "_" + str(x) for x in value["id"]]
        d.append(dict(value["id"].value_counts()))

    c = Counter()
    for dct in d:
        c.update(dct)
    c = dict(c)

    df_list = []
    for k, v in c.items():
        split = k.split("_")
        df_list.append([split[0], split[1], v])
    df = pd.DataFrame(df_list, columns=["Labels", "id", "count"])
    df["id"] = df["id"].str.split(".").str[0]

    info = {}
    for l in sample_data.keys():
        total = df[df["Labels"] == l]["count"].sum()
        info[l] = total

    plt.clf()
    df.groupby(["id", "Labels"]).sum().unstack().plot(
        kind="bar",
        y="count",
        figsize=(9.20, 9.20),
        stacked=True,
        xlabel="Transponders",
        ylabel="Number of samples",
        title=f"Distribution of samples across transponders\n{str(info)}",
    )
    filepath = str(out_dir / filename)
    print(filepath)
    plt.savefig(filepath, bbox_inches="tight")

    # pie chart
    # plt.clf()
    # labels = list(sample_data.keys())
    # sizes = []
    # for k, v in sample_data.items():
    #     sizes.append(len(v))
    # fig1, ax1 = plt.subplots(figsize=(12.80, 7.20))
    # explode = None
    # if 'cedara' in str(out_dir):
    #     explode = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3)
    # ax1.pie(sizes, labels=labels, autopct='%1.1f%%', explode=explode,
    #         shadow=False, startangle=90)
    # ax1.axis('equal')
    # ax1.set_title(f"Famacha transition of samples across herd\n{info}")
    # filepath = str(out_dir / f'pie_{filename}')
    # print(filepath)
    # plt.savefig(filepath, bbox_inches='tight')

    # grid chart
    if grid_enabled:
        max_famacha = (
            np.array([[x[0], x[-1]] for x in sample_data.keys()])
            .flatten()
            .astype(int)
            .max()
        )
        if max_famacha <= 1:
            print(f"no famacha data. max_famacha={max_famacha}")
            return
        mat_label = np.zeros((max_famacha, max_famacha), dtype=object)
        for i in range(mat_label.shape[0]):
            for j in range(mat_label.shape[1]):
                mat_label[i, j] = f"{i+1}To{j+1}"
        mat_label = np.flip(mat_label, axis=0)

        mat_value = np.zeros(mat_label.shape)
        for i in range(mat_value.shape[0]):
            for j in range(mat_value.shape[1]):
                k = mat_label[i, j]
                if k not in sample_data:
                    continue
                mat_value[i, j] = len(sample_data[k])

        yaxis = [f"Famacha {x}" for x in np.arange(1, len(mat_value) + 1)][::-1]

        xaxis = [f"Famacha {x}" for x in np.arange(1, len(mat_value) + 1)]

        fig, ax = plt.subplots()
        im = ax.imshow(mat_value)
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Number of samples", rotation=-90, va="bottom")

        # Show all ticks and label them with the respective list entries
        ax.set_xticks(np.arange(len(xaxis)))
        ax.set_yticks(np.arange(len(yaxis)))
        ax.set_xticklabels(xaxis)
        ax.set_yticklabels(yaxis)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(len(yaxis)):
            for j in range(len(xaxis)):
                text = ax.text(j, i, mat_label[i, j], ha="center", va="center", color="red")

        ax.set_title(f"Famacha transition of samples across herd\n{info}")
        fig.tight_layout()
        filepath = str(out_dir / f"grid_{filename}")
        print(filepath)
        plt.savefig(filepath, bbox_inches="tight")

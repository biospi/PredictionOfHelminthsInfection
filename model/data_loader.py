from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_activity_data(out_dir, filepath, day, class_healthy, class_unhealthy, keep_2_only=True, imputed_days=7,
                       preprocessing_steps=[['QN', 'ANSCOMBE', 'LOG']], hold_out_pct = 0, farm='delmas'):
    print(f"load activity from datasets...{filepath}")
    data_frame = pd.read_csv(filepath, sep=",", header=None, low_memory=False)
    data_frame = data_frame.astype(dtype=float, errors='ignore')  # cast numeric values as float
    data_point_count = data_frame.shape[1]
    hearder = [str(n) for n in range(0, data_point_count)]
    N_META = 4
    hearder[-4] = 'label'
    hearder[-3] = 'id'
    hearder[-2] = 'imputed_days'
    hearder[-1] = 'date'
    data_frame.columns = hearder
    data_frame = data_frame[~np.isnan(data_frame["imputed_days"])]
    mrnn_files = [str(x) for x in list((Path(filepath).parent / "mrnn_windows").glob("*.csv"))]

    if len(mrnn_files) > 0:
        data_frame["mrnn_file"] = mrnn_files

    data_frame = data_frame[data_frame["imputed_days"] <= imputed_days]

    data_frame = data_frame[data_frame.nunique(1) > 10]
    data_frame = data_frame.dropna(subset=data_frame.columns[:-N_META], how='all')
    data_frame = data_frame.dropna()

    if 'ZEROPAD' in preprocessing_steps[0]:
        data_frame = data_frame.fillna(0)
    if 'LINEAR' in preprocessing_steps[0]:
        data_frame.iloc[:, :-N_META] = data_frame.iloc[:, :-N_META].interpolate(axis=1, limit_direction='both')

    #data_frame = data_frame[(data_frame == 0).sum(axis=1) / len(data_frame.columns) <= 0.5]
    # filter with imputed_days count
    # if 'cat' not in filepath:#todo parametarize
    #     data_frame = data_frame[data_frame["imputed_days"] >= day]

    # data_frame = data_frame[data_frame["label"].isin(["1To2", "1To1", "2To2"])]

    # if class_unhealthy is None:
    #     data_frame_labeled = pd.get_dummies(data_frame, columns=["label"])
    #     flabels = [x for x in data_frame_labeled.columns if 'label' in x]
    #     data_frame["target"] = 0
    #     for i, flabel in enumerate(flabels):
    #         data_frame_labeled[flabel] = data_frame_labeled[flabel] * (i + 1)
    #         data_frame["target"] = data_frame["target"] + data_frame_labeled[flabel]
    #     label_series = dict(data_frame[['target', 'label']].drop_duplicates().values)
    #     data_frame = data_frame.drop('label', 1)
    #     return data_frame, N_META, None, None, label_series

    # #1To1 1To2 2To2 2To1
    # new_label = []
    # for v in data_frame["label"].values:
    #     if v in ["1To1"]:
    #         new_label.append("1To1")
    #         continue
    #     new_label.append("2To2")
    #
    # data_frame["label"] = new_label


    # if filter:
    #     if "cedara" in filepath:#todo parametarize
    #         new_label = []
    #         for v in data_frame["label"].values:
    #             if v in ["1To1", "2To2"]:
    #                 new_label.append("1To1")
    #                 continue
    #             if v in ["2To4", "3To4", "1To4", "1To3", "4To5", "2To3"]:
    #                 new_label.append("2To2")
    #                 continue
    #             new_label.append(np.nan)
    #
    #         data_frame["label"] = new_label
    #
    #     if "delmas" in filepath:#todo parametarize
    #         new_label = []
    #         for v in data_frame["label"].values:
    #             if v in ["1To1"]:
    #                 new_label.append("1To1")
    #                 continue
    #             if v in ["2To2"]:
    #                 new_label.append("2To2")
    #                 continue
    #             new_label.append(np.nan)
    #
    #         data_frame["label"] = new_label
    #
    #     if keep_2_only:
    #         data_frame = data_frame.dropna()

    new_label = []
    for v in data_frame["label"].values:
        if v in class_healthy:
            new_label.append("1To1")
            continue
        if v in class_unhealthy:
            new_label.append("2To2")
            continue
        new_label.append(v)

    data_frame["label"] = new_label


    #{4: '2To2', 2: '1To2', 1: '1To1', 3: '2To1'}
    #{'2To2_4': 71, '1To2_2': 47, '1To1_1': 67, '2To1_3': 50}
    #{2: '1To2', 4: '2To2', 3: '2To1', 1: '1To1', 6: '3To2', 8: '4To2', 5: '3To1', 10: '5To2', 7: '3To3', 9: '4To3'}
    # if True:
    #     new_label = []
    #     for v in data_frame["label"].values:
    #         if v in ["2To1"]:
    #             new_label.append("1To1")
    #             continue
    #         if v in ["1To2"]:
    #             new_label.append("2To2")
    #             continue
    #         new_label.append("ignore")
    #     data_frame["label"] = new_label
    # if True:
    #     new_label = []
    #     for v in data_frame["label"].values:
    #         if v in ["2To1", "1To1"]:
    #             new_label.append("1To1")
    #             continue
    #         if v in ["2To2", "1To2"]:
    #             new_label.append("2To2")
    #             continue
    #         new_label.append("ignore")
    #     data_frame["label"] = new_label


    # up_down = True
    # if up_down:
    #     new_label = []
    #     for v in data_frame["label"].values:
    #         split = v.split("To")
    #         a = int(split[0])
    #         b = int(split[1])
    #         # if a == 1 or b == 1:
    #         #     new_label.append("1To1")
    #         #     continue
    #         # if a == b:
    #         #     new_label.append("2To2")
    #         #     continue
    #         if a > b:
    #             new_label.append("1To1")
    #             continue
    #         if a < b:
    #             new_label.append("2To2")
    #             continue
    #         new_label.append(np.nan)
    #     data_frame["label"] = new_label

    #data_frame = data_frame.dropna()
    #data_frame_o = data_frame.copy()
    #print(data_frame)

    # Hot Encode of FAmacha targets and assign integer target to each famacha label
    data_frame_labeled = pd.get_dummies(data_frame, columns=["label"])
    flabels = [x for x in data_frame_labeled.columns if 'label' in x]
    data_frame["target"] = 0
    for i, flabel in enumerate(flabels):
        data_frame_labeled[flabel] = data_frame_labeled[flabel] * (i + 1)
        data_frame["target"] = data_frame["target"] + data_frame_labeled[flabel]


    #store all samples for later testing after binary fitting
    labels = data_frame["label"].drop_duplicates().values
    samples = {}
    for label in labels:
        df = data_frame[data_frame["label"] == label]
        df = df.drop('label', 1)
        samples[label] = df

    plot_samples_distribution(out_dir, samples, f"distrib_all_samples_{farm}.png")

    class_count = {}
    label_series = dict(data_frame[['target', 'label']].drop_duplicates().values)
    label_series_inverse = dict((v, k) for k, v in label_series.items())
    print(label_series_inverse)
    class_healthy = label_series_inverse["1To1"]
    class_unhealthy = label_series_inverse["2To2"]
    print(label_series)
    for k in label_series.keys():
        class_count[label_series[k] + "_" + str(k)] = data_frame[data_frame['target'] == k].shape[0]
    print(class_count)
    # drop label column stored previously, just keep target for ml
    data_frame = data_frame.drop('label', 1)

    print(data_frame)
    # entropy_rows = []
    # for index, row in data_frame.iterrows():
    #     A = row[:-N_META].values.astype(float)
    #     A = np.unique(A)
    #     pA = A / A.sum()
    #     Shannon2 = -np.nansum(pA * np.log2(pA))
    #     entropy_rows.append(len(A))
    #
    # data_frame["e"] = entropy_rows
    # data_frame = data_frame.sort_values(by=['e'], ascending=False)
    # data_frame = data_frame[data_frame["e"] > 150]
    # data_frame = data_frame.drop('e', 1)

    #setup holdout!
    # if hold_out_pct > 0:
    #     class_1_hds = int(data_frame[data_frame['target'] == class_healthy].shape[0] * hold_out_pct/100)
    #     class_2_hds = int(data_frame[data_frame['target'] == class_unhealthy].shape[0] * hold_out_pct / 100)
    #
    #     hould_out_1 = data_frame[data_frame['target'] == class_healthy].sample(n=class_1_hds, random_state=0)
    #     hould_out_2 = data_frame[data_frame['target'] == class_unhealthy].sample(n=class_2_hds, random_state=0)
    #
    #     data_frame = data_frame.drop(hould_out_1.index)
    #     data_frame = data_frame.drop(hould_out_2.index)
    #
    #     samples[label_series[class_healthy]] = hould_out_1
    #     samples[label_series[class_unhealthy]] = hould_out_2
    #
    # data_frame = data_frame[data_frame['target'].isin([class_healthy, class_unhealthy])]

    #plot_samples_distribution(out_dir, samples, f"distrib_hold_out_{Path(filepath).stem}.png")

    return data_frame, N_META, class_healthy, class_unhealthy, label_series, samples


def parse_param_from_filename(file):
    split = file.split("/")[-1].split('.')[0].split('_')
    # activity_delmas_70101200027_dbft_1_1min
    sampling = ""
    days = 0
    farm_id = "farm_id"
    option = ""
    for s in split:
        if 'day' in s:
            days = int(s[0])
            break

    return days, farm_id, option, sampling


def plot_samples_distribution(out_dir, samples, filename):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(samples)
    #bar plot
    d = []
    for key, value in samples.items():
        value['id'] = [key+"_"+str(x) for x in value['id']]
        d.append(dict(value['id'].value_counts()))

    c = Counter()
    for dct in d:
        c.update(dct)
    c = dict(c)

    df_list = []
    for k, v in c.items():
        split = k.split('_')
        df_list.append([split[0], split[1], v])
    df = pd.DataFrame(df_list, columns=["Famacha label", "id", "count"])
    df['id'] = df['id'].str.split('.').str[0]

    info = {}
    for l in samples.keys():
        total = df[df["Famacha label"] == l]["count"].sum()
        info[l] = total

    plt.clf()
    plt.figure(figsize=(12.80, 7.20))
    df.groupby(['id', 'Famacha label']).sum().unstack().plot(kind='bar', y='count',
                                                     stacked=True,
                                                     xlabel="Transponders",
                                                     ylabel="Number of samples",
                                                     title=f"Distribution of samples across transponders\n{str(info)}")
    filepath = str(out_dir / filename)
    print(filepath)
    plt.savefig(filepath, bbox_inches = 'tight')

    # pie chart
    plt.clf()
    labels = list(samples.keys())
    sizes = []
    for k, v in samples.items():
        sizes.append(len(v))
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
            shadow=False, startangle=90)
    ax1.axis('equal')
    ax1.set_title(f"Distribution of usable samples across herd\n{info}")
    filepath = str(out_dir / f'pie_{filename}')
    print(filepath)
    plt.savefig(filepath, bbox_inches='tight')

    #grid chart
    max_famacha = np.array([[x[0], x[-1]] for x in samples.keys()]).flatten().astype(int).max()
    mat_label = np.zeros((max_famacha, max_famacha), dtype=object)
    for i in range(mat_label.shape[0]):
        for j in range(mat_label.shape[1]):
            mat_label[i, j] = f"{i+1}To{j+1}"
    mat_label = np.flip(mat_label, axis=0)

    mat_value = np.zeros(mat_label.shape)
    for i in range(mat_value.shape[0]):
        for j in range(mat_value.shape[1]):
            mat_value[i, j] = len(samples[mat_label[i, j]])

    yaxis = [f"Famacha {x}" for x in np.arange(1, len(mat_value)+1)][::-1]

    xaxis = [f"Famacha {x}" for x in np.arange(1, len(mat_value)+1)]

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
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(yaxis)):
        for j in range(len(xaxis)):
            text = ax.text(j, i, mat_label[i, j],
                           ha="center", va="center", color="green")

    ax.set_title(f"Distribution of usable samples across herd\n{info}")
    fig.tight_layout()
    filepath = str(out_dir / f'grid_{filename}')
    print(filepath)
    plt.savefig(filepath, bbox_inches='tight')




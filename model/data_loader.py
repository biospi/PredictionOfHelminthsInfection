import pandas as pd
import numpy as np


def loadActivityData(filepath, day, class_healthy, class_unhealthy):
    print("load activity from datasets...", filepath)
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
    data_frame = data_frame.fillna(-1)
    # filter with imputed_days count
    data_frame = data_frame[data_frame["imputed_days"] >= day]

    #1To1 1To2 2To2 2To1
    if "cedara" in filepath:
        new_label = []
        for v in data_frame["label"].values:
            if v in ["1To1"]:
                new_label.append("1To1")
                continue
            if v in ["2To4", "3To4", "1To4", "1To3", "4To5", "2To3"]:
                new_label.append("2To2")
                continue
            new_label.append(v)

        data_frame["label"] = new_label

    # up_down = False
    # if up_down:
    #     new_label = []
    #     for v in data_frame["label"].values:
    #
    #         split = v.split("To")
    #         a = int(split[0])
    #         b = int(split[1])
    #
    #         if a == 1 or b == 1:
    #             new_label.append("1To1")
    #             continue
    #         # if a < b:
    #         #     new_label.append("2To2")
    #         #     continue
    #
    #         new_label.append("2To2")
    #
    #     data_frame["label"] = new_label

    data_frame_o = data_frame.copy()
    print(data_frame)

    # Hot Encode of FAmacha targets and assign integer target to each famacha label
    data_frame_labeled = pd.get_dummies(data_frame, columns=["label"])
    flabels = [x for x in data_frame_labeled.columns if 'label' in x]
    data_frame["target"] = 0
    for i, flabel in enumerate(flabels):
        data_frame_labeled[flabel] = data_frame_labeled[flabel] * (i + 1)
        data_frame["target"] = data_frame["target"] + data_frame_labeled[flabel]
    class_count = {}
    label_series = dict(data_frame[['target', 'label']].drop_duplicates().values)
    label_series_inverse = dict((v, k) for k, v in label_series.items())
    class_healthy = label_series_inverse[class_healthy]
    class_unhealthy = label_series_inverse[class_unhealthy]
    print(label_series)
    for k in label_series.keys():
        class_count[label_series[k] + "_" + str(k)] = data_frame[data_frame['target'] == k].shape[0]
    print(class_count)
    # drop label column stored previously, just keep target for ml
    data_frame = data_frame.drop('label', 1)
    print(data_frame)

    return data_frame, N_META, class_healthy, class_unhealthy, label_series


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


def main():
    print("")


if __name__ == "__main__":
    print("")
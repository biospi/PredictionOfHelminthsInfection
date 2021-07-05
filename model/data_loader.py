import pandas as pd
import numpy as np


def loadActivityData(filepath, day):
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

    return data_frame, N_META


def main():
    print("")


if __name__ == "__main__":
    print("")
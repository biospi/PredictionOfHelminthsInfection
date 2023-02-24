import numpy as np
import pandas as pd


def find_samples(df, date, id, label):
    header = df.columns.tolist()
    header[-1] = "date"
    header[-3] = "id"
    header[-4] = "label"
    df.columns = header
    df_sample = df[df["date"] == date]
    df_sample = df_sample[df_sample["label"] == label]
    df_sample = df_sample[df_sample["id"] == id]
    return df_sample.values.flatten()


def main(dataset_mrnn=None, dataset_gain=None, dataset_li=None):
    df_mrnn = pd.read_csv(dataset_mrnn)
    df_gain = pd.read_csv(dataset_gain)
    df_li = pd.read_csv(dataset_li)
    print("loaded data.")
    samples_li = []
    samples_gain = []
    for index, row in df_mrnn.iterrows():
        print(f"{index}/{len(df_mrnn)}...")
        date = row.values[-1]
        id = row.values[-3]
        label = row.values[-4]
        sample_raw_li = find_samples(df_li, date, id, label)
        samples_li.append(sample_raw_li)
        sample_gain = find_samples(df_gain, date, id, label)
        samples_gain.append(sample_gain)
    df_gain_ = pd.DataFrame(samples_gain)
    df_raw_ = pd.DataFrame(samples_li)
    print(0)


if __name__ == "__main__":
    cedara_path_mrnn = "E:/thesis/datasets/cedara/cedara_datasetmrnn7_23/activity_farmid_dbft_7_1min.csv"
    cedara_path_gain = "E:/thesis/datasets/cedara/cedara_dataset_1_gain_172/activity_farmid_dbft_7_1min.csv"
    cedara_path_li = "E:/thesis/datasets/cedara/cedara_dataset_1_li_172/activity_farmid_dbft_7_1min.csv"
    main(cedara_path_mrnn, cedara_path_gain, cedara_path_li)
    
    delmas_path_mrnn = "E:/thesis/datasets/delmas/delmas_dataset4_mrnn_7day/activity_farmid_dbft_7_1min.csv"
    delmas_path_gain = "E:/thesis/datasets/delmas/delmas_dataset_1_gain_66/activity_farmid_dbft_7_1min.csv"
    delmas_path_li = "E:/thesis/datasets/delmas/delmas_dataset_1_li_66/activity_farmid_dbft_7_1min.csv"
    main(delmas_path_mrnn, delmas_path_gain, delmas_path_li)
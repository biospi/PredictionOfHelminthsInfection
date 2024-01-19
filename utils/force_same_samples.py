import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

from utils.Utils import anscombe


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


def export_dataset(path, dataset):
    dataset_name = Path(path).parent.stem + "_fixed"
    dataset_folder = Path(path).parent.parent / dataset_name
    dataset_folder.mkdir(parents=True, exist_ok=True)
    filepath = dataset_folder / Path(path).name
    print(filepath)
    dataset.to_csv(filepath, index=False, header=None)
    orig_json = Path(path).parent / f"{Path(path).stem}.json"
    copy_to = dataset_folder / f"{Path(path).stem}.json"
    print(copy_to)
    shutil.copy(orig_json, copy_to)


def main(dataset_mrnn=None, dataset_gain=None, dataset_li=None, filename = "heatmap_samples_imp.html"):
    df_mrnn = pd.read_csv(dataset_mrnn, header=None)
    df_mrnn = df_mrnn.dropna()
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


    df_gain_ = pd.DataFrame(samples_gain, columns=df_mrnn.columns)
    df_li_ = pd.DataFrame(samples_li, columns=df_mrnn.columns)

    export_dataset(dataset_gain, df_gain_)
    export_dataset(dataset_li, df_li_)

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(f"MRNN Samples ({df_mrnn.shape[0]})", f"{df_gain_.iloc[:, -4].value_counts().to_dict()}<br>GAIN Samples ({df_mrnn.shape[0]})", f"LI Samples ({df_mrnn.shape[0]})"),
        y_title="",
        x_title="Time (1 min bins)",
    )
    for i, df in enumerate([df_mrnn, df_gain_, df_li_]):
        header = [x for x in df.columns]
        header[-4] = "label"
        df.columns = header
        df = df.sort_values("label")
        yaxis_label = df.iloc[:, -4].values
        yaxis_label = yaxis_label + "_" + np.arange(0, len(yaxis_label)).astype(str)
        df = df.iloc[:, :-4]
        xaxix_label = np.arange(0, df.shape[0], 1)
        matrix = df.values
        matrix[0][0] = 0 #prevent plotly axis bug when matrix only contains nan
        trace = go.Heatmap(
            z=np.log(anscombe(matrix)).astype(float),
            x=xaxix_label,
            y=yaxis_label,
            #colorbar=dict(x=1 + i / 30, title=f"fig:{i}"),
            colorscale="Viridis",
            showscale=False
        )
        fig.append_trace(trace, row=1, col=i + 1)

    out_dir = Path("heatmaps")
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / filename
    print(output)
    fig.write_html(str(output))


if __name__ == "__main__":
    delmas_path_mrnn = "E:/thesis/datasets/delmas/delmas_dataset4_mrnn_7day/activity_farmid_dbft_7_1min.csv"
    delmas_path_gain = "E:/thesis/datasets/delmas/delmas_dataset_1_gain_66_no_filter/activity_farmid_dbft_7_1min.csv"
    delmas_path_li = "E:/thesis/datasets/delmas/delmas_dataset_1_li_66_no_filter/activity_farmid_dbft_7_1min.csv"
    main(delmas_path_mrnn, delmas_path_gain, delmas_path_li, filename = "delmas_heatmap_samples_imp.html")

    cedara_path_mrnn = "E:/thesis/datasets/cedara/cedara_datasetmrnn7_23/activity_farmid_dbft_7_1min.csv"
    cedara_path_gain = "E:/thesis/datasets/cedara/cedara_dataset_1_gain_172_no_filter/activity_farmid_dbft_7_1min.csv"
    cedara_path_li = "E:/thesis/datasets/cedara/cedara_dataset_1_li_172_no_filter/activity_farmid_dbft_7_1min.csv"
    main(cedara_path_mrnn, cedara_path_gain, cedara_path_li, filename = "cedara_heatmap_samples_imp.html")


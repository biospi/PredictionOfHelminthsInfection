import argparse
import glob
import pathlib
from sys import exit
import matplotlib
import typer

from model.data_loader import load_activity_data, parse_param_from_filename
from preprocessing.preprocessing import applyPreprocessingSteps
from utils.Utils import getXY
from utils.visualisation import plot_2d_space

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn import cluster, datasets
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

from pathlib import Path
from utils._anscombe import Anscombe, Log
from utils._normalisation import QuotientNormalizer
from typing import List
import tqdm
import tqdm.asyncio
import datashader
import colorcet
import holoviews
import matplotlib
import umap
import umap.plot
import sklearn.datasets



def preprocess(X, out_dir):
    X = QuotientNormalizer(out_dir=out_dir + "/" + "unsupervised").transform(X)
    X = Anscombe().transform(X)
    X = Log().transform(X)
    return X


def main(
    output_dir: Path = typer.Option(
        ..., exists=False, file_okay=False, dir_okay=True, resolve_path=True
    ),
    dataset_folder: Path = typer.Option(
        ..., exists=True, file_okay=False, dir_okay=True, resolve_path=True
    ),
    steps: List[str] = ["QN", "ANSCOMBE", "LOG"]
):
    """This script use unsupervised learning technique on the data\n
    Args:\n
        output_dir: Output directory
        dataset_folder: Dataset input directory
    """
    files = glob.glob(str(dataset_folder / "*.csv"))  # find datset files
    print("found %d files." % len(files))
    print(files)

    for file in files:
        days, farm_id, option, sampling = parse_param_from_filename(file)
        print(f"loading dataset file {file} ...")
        (
            data_frame,
            N_META,
            class_healthy_target,
            class_unhealthy_target,
            label_series,
        ) = load_activity_data(file, days, None, None)

        print(data_frame)
        data_frame_o = data_frame.copy()

        data_frame = applyPreprocessingSteps(
            days,
            None,
            None,
            None,
            None,
            None,
            data_frame.copy(),
            N_META,
            output_dir,
            steps,
            "class_healthy_label",
            "class_unhealthy_label",
            class_healthy_target,
            class_unhealthy_target,
            clf_name="SVM",
            n_scales=10,
            farm_name="FARMS",
            keep_meta=False,
        )
        print(data_frame)
        y = [str(x).split('.')[0] for x in data_frame["target"].values.flatten()]
        #y = y.astype(int)
        X = data_frame[data_frame.columns[0: data_frame.shape[1] - 1]].values

        mapper = umap.UMAP().fit(X)
        hover_data = pd.DataFrame({'index': np.arange(X.shape[0]), 'label': np.array(y).astype(int)})

        hover_data['item'] = hover_data.label.map(label_series)
        hover_data["animal_id"] = data_frame_o["id"].values
        hover_data["date"] = data_frame_o["date"].values

        #hover_data['item'] = hover_data.label.map(data)

        filename = "umap.html"
        filepath = output_dir / filename

        umap.plot.output_file(str(filepath))

        p = umap.plot.interactive(mapper, labels=y, point_size=10, hover_data=hover_data, width=1000, height=1000)
        umap.plot.show(p)

        # plot_2d_space(X, y, filepath, label_series, title="Umap")

    # df_gt, n_meta = load_activity_data(files[0], 1, 2)
    # X_gt = df_gt.iloc[:, :-n_meta].values
    # # X_gt = preprocess(X_gt, output_dir)
    # # X_gt_pca = PCA(n_components=pca_dim).fit_transform(X_gt)
    # y_gt = df_gt["label"].values
    # y = []
    # for v in y_gt:
    #     if v in ["1To1"]:
    #         y.append(0)
    #         continue
    #     # if v in ["2To4", "3To4", "1To4", "1To3", "4To5", "2To3"]:
    #     #     y.append(1)
    #     #     continue
    #     y.append(2)
    # y = np.array(y)
    #
    # # dataSet = np.array(y_gt.tolist(), dtype='U21'),
    # # lut = np.sort(np.unique(dataSet))
    # # ind = np.searchsorted(lut, dataSet)[0]
    # y_gt = np.unique(y_gt)

    # all_samples = []
    # print("building datasets...")
    # # cpt = 0
    # for file in tqdm(files):
    #     samples = get_samples(file)
    #     all_samples.extend(samples)
    #     # cpt += 1
    #     # if cpt > 2:
    #     #     break
    #
    # df = pd.DataFrame(all_samples)
    # print(df)
    # X = df.values
    # X = preprocess(X, output_dir)
    # X_pca = PCA(n_components=pca_dim).fit_transform(X)

    # find_clusters("KMEAN clusters PCA(3)", output_dir, X_pca, X_gt_pca, y, y_gt)
    # findClusters("KMEAN clusters", output_dir, X)
    # findClusters("KMEAN clusters fit on all features, scatter PCA(3)", output_dir, X)


def get_samples(file):
    # print("load activity from datasets...", file)
    data_frame = pd.read_csv(file, sep=",", low_memory=False)
    data_frame = data_frame.astype(dtype=float, errors="ignore")
    # print(data_frame)
    data = data_frame["first_sensor_value_gain"].values
    n = 1440
    days = [data[i : i + n] for i in range(0, len(data), n)]
    samples = []
    for day in days:
        if len(day) != n:
            print("invalid sample size!")
            continue
        if np.isnan(day).any():
            continue
        samples.append(day)
    return samples


def find_clusters(title, out_dir, X, X_gt_pca, y, labels, n_clusters=2):
    k_means = cluster.KMeans(n_clusters=n_clusters, n_jobs=-1)
    k_means.fit(X)
    y_kmeans = k_means.predict(X)

    fig, ax = plt.subplots(figsize=(12.20, 7.20))
    ax = fig.add_subplot(111, projection="3d")

    # ax.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], X[y_kmeans == 0, 2], marker='o', color='tab:blue', label='Class0 (Healthy)')
    # ax.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], X[y_kmeans == 1, 2], marker='s', color='tab:red', label='Class1 (Unhealthy)')

    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
        "b",
        "g",
        "r",
        "c",
        "m",
        "y",
        "k",
        "w",
        "pink",
    ]

    for i in np.unique(y)[:-1]:
        ax.scatter(
            X_gt_pca[y == i, 0],
            X_gt_pca[y == i, 1],
            X_gt_pca[y == i, 2],
            marker="o",
            color=colors[i],
            edgecolor="black",
            label=labels[i],
        )

    # centers = k_means.cluster_centers_
    # ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='black', alpha=0.5)
    ax.set(xlabel="PCA component 1", ylabel="PCA component 2", zlabel="PCA component 3")
    ax.legend(loc="lower right")
    ax.view_init(30, 60)

    plt.title(title)
    ttl = ax.title
    ttl.set_position([0.57, 0.97])
    path = "%s/" % (out_dir)
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    filename = "%s.png" % (title)
    final_path = "%s/%s" % (path, filename)
    print(final_path)
    try:
        plt.savefig(final_path, bbox_inches="tight")
    except FileNotFoundError as e:
        print(e)
        exit()
    plt.show()
    # plt.close()
    # plt.close()
    # fig.clear()


def test():
    X, y = datasets.load_iris(return_X_y=True)
    find_clusters(X, y)


if __name__ == "__main__":
    typer.run(main)
    # print("********************************************************************")
    # print("*                    ML PIPELINE UNSUPERVISED                      *")
    # print("********************************************************************")
    # parser = argparse.ArgumentParser()
    # parser.add_argument("output_dir", help="output directory", type=str)
    # parser.add_argument("dataset_folder", help="dataset input directory", type=str)
    # parser.add_argument(
    #     "gt_dataset_folder", help="ground truth dataset input directory", type=str
    # )
    # parser.add_argument("--pca_dim", help="PCA components", type=int, default=3)
    # args = parser.parse_args()
    #
    # output_dir = args.output_dir
    # dataset_folder = args.dataset_folder
    # gt_dataset_folder = args.gt_dataset_folder
    # pca_dim = args.pca_dim
    #
    # main(output_dir, gt_dataset_folder, dataset_folder, pca_dim)

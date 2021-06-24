import argparse
import glob
import pathlib
from sys import exit
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn import cluster, datasets
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

from utils._anscombe import Anscombe, Log
from utils._normalisation import QuotientNormalizer



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
    # new_label = []
    # for v in data_frame["label"].values:
    #     if v in ["1To1"]:
    #         new_label.append("1To1")
    #         continue
    #     if v in ["2To4", "3To4", "1To4", "1To3", "4To5", "2To3"]:
    #         new_label.append("1To2")
    #         continue
    #     new_label.append(v)
    #
    # data_frame["label"] = new_label
    return data_frame, N_META


def preprocess(X, out_dir):
    X = QuotientNormalizer(out_dir=out_dir + "/" + "unsupervised").transform(X)
    X = Anscombe().transform(X)
    X = Log().transform(X)
    return X


def main(output_dir, gt_dataset_folder, dataset_folder, pca_dim):
    print("output_dir", output_dir)
    print("dataset_folder", dataset_folder)
    print("gt_dataset_folder", gt_dataset_folder)

    files = glob.glob(dataset_folder + "/*.csv")  # find dataset files
    files = [file.replace("\\", '/') for file in files]
    print("found %d files." % len(files))
    print(files)

    files_gt = glob.glob(gt_dataset_folder + "/*.csv")  # find dataset files
    files_gt = [file.replace("\\", '/') for file in files_gt]
    print("found %d gt files." % len(files_gt))
    print(files_gt)

    df_gt, n_meta = loadActivityData(files_gt[0], 1)
    X_gt = df_gt.iloc[:, :-n_meta].values
    X_gt = preprocess(X_gt, output_dir)
    X_gt_pca = PCA(n_components=pca_dim).fit_transform(X_gt)
    y_gt = df_gt["label"].values
    y = []
    for v in y_gt:
        if v in ["1To1"]:
            y.append(0)
            continue
        if v in ["2To4", "3To4", "1To4", "1To3", "4To5", "2To3"]:
            y.append(1)
            continue
        y.append(2)
    y = np.array(y)

    # dataSet = np.array(y_gt.tolist(), dtype='U21'),
    # lut = np.sort(np.unique(dataSet))
    # ind = np.searchsorted(lut, dataSet)[0]
    y_gt = np.unique(y_gt)

    all_samples = []
    print("building datasets...")
    cpt = 0
    for file in tqdm(files):
        samples = getSamples(file)
        all_samples.extend(samples)
        cpt += 1
        if cpt > 2:
            break

    df = pd.DataFrame(all_samples)
    print(df)
    X = df.values
    X = preprocess(X, output_dir)
    X_pca = PCA(n_components=pca_dim).fit_transform(X)

    findClusters("KMEAN clusters PCA(3)", output_dir, X_pca, X_gt_pca, y, y_gt)
    # findClusters("KMEAN clusters", output_dir, X)
    # findClusters("KMEAN clusters fit on all features, scatter PCA(3)", output_dir, X)


def getSamples(file):
    #print("load activity from datasets...", file)
    data_frame = pd.read_csv(file, sep=",", low_memory=False)
    data_frame = data_frame.astype(dtype=float, errors='ignore')
    #print(data_frame)
    data = data_frame["first_sensor_value_gain"].values
    n = 1440
    days = [data[i:i + n] for i in range(0, len(data), n)]
    samples = []
    for day in days:
        if len(day) != n:
            print("invalid sample size!")
            continue
        if np.isnan(day).any():
            continue
        samples.append(day)
    return samples


def findClusters(title, out_dir, X, X_gt_pca, y, labels, n_clusters=2):
    k_means = cluster.KMeans(n_clusters=n_clusters, n_jobs=-1)
    k_means.fit(X)
    y_kmeans = k_means.predict(X)

    fig, ax = plt.subplots(figsize=(12.20, 7.20))
    ax = fig.add_subplot(111, projection='3d')

    # ax.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], X[y_kmeans == 0, 2], marker='o', color='tab:blue', label='Class0 (Healthy)')
    # ax.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], X[y_kmeans == 1, 2], marker='s', color='tab:red', label='Class1 (Unhealthy)')

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'pink']

    for i in np.unique(y)[:-1]:
        ax.scatter(X_gt_pca[y == i, 0], X_gt_pca[y == i, 1], X_gt_pca[y == i, 2],  marker='o', color=colors[i], edgecolor="black", label=labels[i])

    # centers = k_means.cluster_centers_
    # ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='black', alpha=0.5)
    ax.set(xlabel="PCA component 1", ylabel="PCA component 2", zlabel="PCA component 3")
    ax.legend(loc="lower right")
    ax.view_init(30, 60)

    plt.title(title)
    ttl = ax.title
    ttl.set_position([.57, 0.97])
    path = "%s/" % (out_dir)
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    filename = "%s.png" % (title)
    final_path = '%s/%s' % (path, filename)
    print(final_path)
    try:
        plt.savefig(final_path, bbox_inches='tight')
    except FileNotFoundError as e:
        print(e)
        exit()
    plt.show()
    # plt.close()
    # plt.close()
    # fig.clear()


def test():
    X, y = datasets.load_iris(return_X_y=True)
    findClusters(X, y)


if __name__ == "__main__":
    print("********************************************************************")
    print("*                    ML PIPELINE UNSUPERVISED                      *")
    print("********************************************************************")
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir', help='output directory', type=str)
    parser.add_argument('dataset_folder', help='dataset input directory', type=str)
    parser.add_argument('gt_dataset_folder', help='ground truth dataset input directory', type=str)
    parser.add_argument('--pca_dim', help='PCA components', type=int, default=3)
    args = parser.parse_args()

    output_dir = args.output_dir
    dataset_folder = args.dataset_folder
    gt_dataset_folder = args.gt_dataset_folder
    pca_dim = args.pca_dim

    main(output_dir, gt_dataset_folder, dataset_folder, pca_dim)

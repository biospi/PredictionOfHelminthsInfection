import numpy as np
import pandas as pd
from pathlib import Path
import json
import matplotlib.pyplot as plt


def main(folder_path):
    files = list(folder_path.glob("**/*results.json"))
    print(files)

    # files_ = []
    # for f in files:
    #     if "QN_STD_rbf" not in str(f):
    #         continue
    #     files_.append(f)

    # files = files_
    full_data = []
    for i, file in enumerate(files):
        print(file)
        print(f"{i}/{len(files)}")
        data = json.load(open(file))
        #print(data)
        config = file.parent.parent.stem
        aucs_test = [x['auc'] for x in data['SVC_rbf_results']]
        aucs_train = [x['auc_train'] for x in data['SVC_rbf_results']]
        aucs_test_median = np.median(aucs_test)
        aucs_train_median = np.median(aucs_train)
        training_shape = data['SVC_rbf_results'][0]['training_shape']
        testing_shape = data['SVC_rbf_results'][0]['testing_shape']
        n_c = file.parent.parent.parent.stem
        dataset = file.parent.parent.parent.parent.stem
        full_data.append([config, aucs_test, aucs_train, aucs_test_median, aucs_train_median, training_shape[0], testing_shape[0], n_c, dataset])
    df_data = pd.DataFrame(full_data, columns=["config", "auc_test", "auc_train", "auc_test_median", "auc_train_median", "training_shape", "testing_shape", "n_c", "dataset"])
    #df_data['n_c'] = df_data['n_c'].astype(float)
    #df_data = df_data[df_data["dataset"] == "003__0_00100__120"]
    #df_data = df_data[df_data['n_c'] < 45]
    df_data["n_peaks"] = [int(x.split('__')[1]) for x in df_data['n_c'].values]
    df_data["thresh"] = [ int(x.split('__')[0]) for x in df_data['n_c'].values]
    dfs_data = [group for _, group in df_data.groupby(df_data["config"])]

    fig, ax = plt.subplots(figsize=(19.20, 10.80))
    for df_data in dfs_data:

        for peak in df_data["n_peaks"].unique():
            df_plot = df_data[df_data["n_peaks"] == peak]
            df_plot = df_plot.sort_values(by='training_shape')

            x_axis = df_plot["training_shape"]
            y_axis = df_plot["auc_test_median"]
            y_axis_ = df_plot["auc_train_median"]
            config = df_plot["config"].values[0]
            if "LOG" in config:
                continue
            ax.plot(x_axis, y_axis, lw=2, alpha=0.5, linestyle='-', marker='x', label=f"Test peak={peak}_config={config}")
            ax.plot(x_axis, y_axis_, lw=2, alpha=0.5, linestyle='--', marker='o', label=f"Train peak={peak}_config={config}")
    ax.set_xlabel("Samples in training fold")
    ax.set_ylabel("Median AUC")
    ax.set_title(f"Auc evolution with increasing dataset sample size")
    ax.legend(loc="upper right")
    final_path = folder_path / f"auc_test_train_{config}.png"
    print(final_path)
    fig.savefig(final_path)
    plt.close(fig)

    for df_data in dfs_data:
        fig, ax = plt.subplots(figsize=(19.20, 10.80))
        for thresh in df_data["thresh"].unique():
            df_plot = df_data[df_data["thresh"] == thresh]
            df_plot = df_plot.sort_values(by='training_shape')
            x_axis = df_plot["n_peaks"]
            y_axis = df_plot["auc_test_median"]
            y_axis_ = df_plot["auc_train_median"]
            config = df_plot["config"].values[0]
            ax.plot(x_axis, y_axis, lw=2, alpha=0.5, linestyle='-', marker='x', label=f"Test thresh={thresh}_config={config}")
            ax.plot(x_axis, y_axis_, lw=2, alpha=0.5, linestyle='--', marker='o', label=f"Train thresh={thresh}_config={config}")
        ax.set_xlabel("Samples in training fold")
        ax.set_ylabel("Median AUC")
        ax.set_title(f"Auc evolution with increasing number of peaks")
        ax.legend(loc="upper right")
        final_path = folder_path / f"auc_npeaks_{thresh}.png"
        print(final_path)
        fig.savefig(final_path)
        plt.close(fig)


if __name__ == "__main__":
    main(Path("E:/Cats/bluepebble3/ml_all"))
    #main(Path("E:/Cats/bluepebble2"))
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
    df_data['n_c'] = df_data['n_c'].astype(float)
    df_data = df_data[df_data['n_c'] < 45]
    df_data = df_data[df_data["dataset"] == "003__0_00100__120"]
    dfs_data = [group for _, group in df_data.groupby(df_data["config"])]

    fig, ax = plt.subplots(figsize=(12.80, 7.20))
    for df_data in dfs_data:

        df_data = df_data.sort_values(by='n_c')

        x_axis = df_data["n_c"]
        y_axis = df_data["auc_test_median"]
        config = df_data["config"].values[0]
        ax.plot(x_axis, y_axis, lw=2, alpha=0.5, linestyle='--', marker='x', label=config)
        ax.set_xlabel("Cats in training fold")
        ax.set_ylabel("Median AUC")
        ax.set_title(f"Auc evolution with increasing cats in training fold config={config}")
        ax.legend()
    final_path = folder_path / f"auc_{config}.png"
    print(final_path)
    fig.savefig(final_path)
    plt.close(fig)


if __name__ == "__main__":
    main(Path("E:/Cats/bluepebble2"))
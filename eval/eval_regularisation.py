from pathlib import Path
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize


class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def plot_heatmap(df, col, out_dir, title=""):
    scores = df[col].values
    scores = np.array(scores).reshape(len(df["C"].unique()), len(df["gamma"].unique()))
    #plt.figure(figsize=(8, 6))
    #plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    fig, ax = plt.subplots()
    im = ax.imshow(scores, interpolation='nearest')
    # im = ax.imshow(scores, interpolation='nearest',
    #            norm=MidpointNormalize(vmin=-.2, midpoint=0.5))
    ax.set_xlabel('gamma')
    ax.set_ylabel('C')
    fig.colorbar(im)
    ax.set_xticks(np.arange(len(df["gamma"].unique())),
               [np.format_float_scientific(i, 1) for i in df["gamma"].unique()], rotation=45)
    ax.set_yticks(np.arange(len(df["C"].unique())),
               [np.format_float_scientific(i, ) for i in df["C"].unique()])
    ax.set_title(f'Regularisation AUC\n{title}')
    fig.tight_layout()
    fig.show()
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"heatmap_{col}_{title}.png".replace(":", "_").replace(" ", "_")
    filepath = out_dir / filename
    print(filepath)
    fig.savefig(filepath)


def plot_fig(df, col, out_dir, title=""):
    scores = df[col].values
    scores = np.array(scores).reshape(len(df["C"].unique()), len(df["gamma"].unique()))
    Cs = df["C"].unique()
    Gammas = df["gamma"].unique()
    fig, ax = plt.subplots()
    for ind, i in enumerate(Cs):
        ax.plot(Gammas, scores[ind], label="C: " + str(i))
    ax.set_title(title)
    ax.set_xscale('log')
    ax.legend()
    ax.set_xlabel("Gamma")
    ax.set_ylabel("Median AUC")
    fig.tight_layout()
    fig.show()
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"plot_{col}_{title}.png".replace(":", "_").replace(" ", "_")
    filepath = out_dir / filename
    print(filepath)
    fig.savefig(filepath)


def main(data_dir, out_dir):
    folders = [x for x in data_dir.glob("*/*/*/*") if x.is_dir()]
    results = []
    for i, item in enumerate(folders):
        if "fold_data" not in str(item):
            continue
        print(item)
        paths = np.array(list(item.glob("*.json")))
        if len(paths) == 0:
            return
        fold_auc_test = []
        fold_auc_train = []
        for filepath in paths:
            with open(filepath, "r") as fp:
                try:
                    res = json.load(fp)
                    gamma = res["gamma"]
                    C = res["c"]
                    auc_test = res["auc"]
                    if auc_test < 0.5:
                        auc_test = 0.5
                    auc_train = res["auc_train"]
                    if auc_train < 0.5:
                        auc_train = 0.5
                    fold_auc_test.append(auc_test)
                    fold_auc_train.append(auc_train)
                except Exception as e:
                    print(e)
                    return
        results.append(
            {
                "C": C,
                "gamma": gamma,
                "fold_median_auc_test": np.median(fold_auc_test),
                "fold_median_auc_train": np.median(fold_auc_train),
            }
        )
        print()
    df = pd.DataFrame(results)
    df = df.sort_values(["C", "gamma"])

    dfs = []
    for c in df["C"].unique():
        df_ = df[df["C"] == c]
        df_ = df_[df_["gamma"].isin(df[df["C"] == 1]["gamma"].tolist())]
        print(c, len(df_["gamma"].unique()))
        dfs.append(df_)

    df = pd.concat(dfs)

    plot_fig(
        df,
        "fold_median_auc_test",
        out_dir,
        f"GridSearch Testing model:{filepath.parent.parent.parent.parent.name}",
    )
    plot_fig(
        df,
        "fold_median_auc_train",
        out_dir,
        f"GridSearch Training model:{filepath.parent.parent.parent.parent.name}",
    )

    plot_heatmap(
        df,
        "fold_median_auc_test",
        out_dir,
        f"GridSearch Testing model:{filepath.parent.parent.parent.parent.name}",
    )

    plot_heatmap(
        df,
        "fold_median_auc_train",
        out_dir,
        f"GridSearch Training model:{filepath.parent.parent.parent.parent.name}",
    )


if __name__ == "__main__":
    main(Path("E:/thesis/reg/regularisation/delmas/main_experiment"), Path("E:/thesis/reg/regularisation/delmas/eval/"))
    main(Path("E:/thesis/reg/regularisation/cedara/main_experiment"), Path("E:/thesis/reg/regularisation/cedara/eval/"))
    #main("E:/thesis_Aug24_regularisation/main_experiment")

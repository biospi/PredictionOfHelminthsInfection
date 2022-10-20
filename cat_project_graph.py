from pathlib import Path, PurePath
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import typer
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import matplotlib.cm as cm

DEFAULT_PLOTLY_COLORS = [
    "rgb(31, 119, 180)",
    "rgb(255, 127, 14)",
    "rgb(44, 160, 44)",
    "rgb(214, 39, 40)",
    "rgb(148, 103, 189)",
    "rgb(140, 86, 75)",
    "rgb(227, 119, 194)",
    "rgb(127, 127, 127)",
    "rgb(188, 189, 34)",
    "rgb(23, 190, 207)",
]


def mean_confidence_interval(x):
    # boot_median = [np.median(np.random.choice(x, len(x))) for _ in range(iteration)]
    x.values.sort()
    lo_x_boot = np.percentile(x, 2.5)
    hi_x_boot = np.percentile(x, 97.5)
    # print(lo_x_boot, hi_x_boot)
    return lo_x_boot, hi_x_boot


def local_run(
    input_dir=Path("E:/Cats/article/ml_build_permutations_qnf_final2"),
    out_dir=Path("E:/Cats/article/ml_build_permutations_qnf_final2"),
):
    main(input_dir, out_dir)


def main(
    input_dir: Path = typer.Option(
        ..., exists=True, file_okay=False, dir_okay=True, resolve_path=True
    ),
    out_dir: Path = typer.Option(
        ..., exists=True, file_okay=False, dir_okay=True, resolve_path=True
    ),
):
    data = []
    for p in input_dir.rglob("*.json"):
        if "proba" in str(p) or "fold_data" in str(p):
            continue
        split = list(PurePath(p).parts)
        data.append(split + [p])

    df = pd.DataFrame(data)

    thresh_list = []
    median_auc_list = []
    auc_list = []
    n_samples = []
    window_size_list = []
    p_steps_list = []
    n_peaks = []

    for index, row in df.iterrows():
        res_file_path = row[8]
        try:
            results = json.load(open(res_file_path))
        except Exception as e:
            print(e)
            continue
        p_steps_list.append(row[5])
        window_size_list.append(int(row[4].split("_")[-1]))
        thresh = row[4].split("__")[-2]
        t_v = thresh.replace("_", ".")
        thresh_float = float(t_v)
        n_peaks.append(int(row[4].split("__")[1]))
        print(res_file_path)

        clf_res = results[list(results.keys())[0]]
        aucs = []
        all_y = []
        all_probs = []
        for item in clf_res:
            if "auc" not in item:
                continue
            # auc = item["auc"]
            # aucs.append(auc)

            all_y.extend(item["y_test"])
            all_probs.extend(np.array(item["y_pred_proba_test"])[:, 1])

            training_shape = len(item["ids_train"])
            testing_shape = len(item["ids_test"])

        all_y = np.array(all_y)
        all_probs = np.array(all_probs)
        fpr, tpr, thresholds = roc_curve(all_y, all_probs)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        auc_list.append(aucs)
        median_auc = np.median(aucs)
        thresh_list.append(thresh_float)
        median_auc_list.append(median_auc)
        n_samples.append(training_shape + testing_shape)

    df_data = pd.DataFrame(
        {
            "auc_list": auc_list,
            "median_auc": median_auc_list,
            "thresh_list": thresh_list,
            "n_samples": n_samples,
            "p_steps_list": p_steps_list,
            "window_size_list": window_size_list,
            "n_peaks": n_peaks,
        }
    )

    print(df_data)
    fig, ax1 = plt.subplots(figsize=(7.20, 7.20))
    ax2 = ax1.twinx()
    dfs = [group for _, group in df_data.groupby(['p_steps_list'])]

    ax2.bar(
        [1, 2, 3, 4, 5, 6],
        [520, 4680, 37440, 52000, 52000, 52000],
        color="grey",
        label="n samples",
        alpha=0.4,
        width=0.2
    )

    for i, df in enumerate(dfs):
        dfs_ = [group for _, group in df.groupby(['window_size_list'])]
        for df_ in dfs_:
            # df_ = df_[df_['n_peaks'] <= 4]
            # if 'linear' in df_['p_steps_list'].tolist()[0]:
            #     continue
            if 'linear' in df_['p_steps_list'].tolist()[0]:
                continue
            print(df_['p_steps_list'].tolist()[0])
            #
            # if len(df_["median_auc"]) != 4:
            #     continue
            print(df_["n_samples"])
            ax1.plot(
                df_["n_peaks"],
                df_["median_auc"],
                label=f"Window size={df_['window_size_list'].tolist()[0]*2} sec | {'>'.join(df_['p_steps_list'].tolist()[0].split('_')[4:])}",
                marker="x"
            )
    ax1.axhline(y=0.5, color='black', linestyle='--')
    fig.suptitle("Evolution of AUC with N peak increase")
    ax1.set_xlabel("Number of peaks")
    ax1.set_ylabel("Mean AUC")
    ax2.set_ylabel("Number of samples(high activity peak window)")
    #plt.legend()
    ax1.legend(loc="lower right").set_visible(True)
    ax2.legend(loc="upper left").set_visible(True)
    fig.tight_layout()
    filename = f"auc_per_npeak.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    filepath = out_dir / filename
    print(filepath)
    fig.savefig(filepath)


if __name__ == "__main__":
    local_run()
    # typer.run(main)

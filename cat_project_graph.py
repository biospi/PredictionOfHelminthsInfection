from pathlib import Path, PurePath

import matplotlib
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import typer
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.legend_handler import HandlerBase
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score
from sklearn.metrics import PrecisionRecallDisplay

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

class AnyObjectHandler(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       x0, y0, width, height, fontsize, trans):
        l1 = plt.Line2D([x0,y0+width], [0.7*height,0.7*height],
                           linestyle=orig_handle[1], color=orig_handle[0])
        l2 = plt.Line2D([x0,y0+width], [0.3*height,0.3*height],
                           color=orig_handle[0])
        return [l1, l2]


def mean_confidence_interval(x):
    # boot_median = [np.median(np.random.choice(x, len(x))) for _ in range(iteration)]
    x.values.sort()
    lo_x_boot = np.percentile(x, 2.5)
    hi_x_boot = np.percentile(x, 97.5)
    # print(lo_x_boot, hi_x_boot)
    return lo_x_boot, hi_x_boot


def local_run(
    input_dir=Path("E:/Cats/article/ml_build_permutations_qnf_final3"),
    out_dir=Path("E:/Cats/article/ml_build_permutations_qnf_final3"),
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
    median_auc_test_list = []
    median_auc_train_list = []
    auc_test_list = []
    auc_train_list = []
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
        steps = row[5]
        p_steps_list.append(steps)
        windowsize = int(row[4].split("_")[-1])
        window_size_list.append(windowsize)
        thresh = row[4].split("__")[-2]
        t_v = thresh.replace("_", ".")
        thresh_float = float(t_v)
        npeak = int(row[4].split("__")[1])
        n_peaks.append(npeak)
        print(res_file_path)

        clf_res = results[list(results.keys())[0]]
        aucs_test = []
        aucs_train = []
        precision_test = []
        precision_train = []
        all_y_test = []
        all_probs_test = []
        all_y_train = []
        all_probs_train = []
        for item in clf_res:
            if "auc" not in item:
                continue
            # auc = item["auc"]
            # aucs.append(auc)

            all_y_test.extend(item["y_test"])
            all_probs_test.extend(np.array(item["y_pred_proba_test"]))

            all_y_train.extend(item["y_train"])
            all_probs_train.extend(np.array(item["y_pred_proba_train"]))

            training_shape = len(item["ids_train"])
            testing_shape = len(item["ids_test"])

        all_y_test = np.array(all_y_test)
        all_probs_test = np.array(all_probs_test)
        fpr, tpr, thresholds = roc_curve(all_y_test, all_probs_test)
        roc_auc = auc(fpr, tpr)
        aucs_test.append(roc_auc)


        display = PrecisionRecallDisplay.from_predictions(all_y_test, all_probs_test, name="LinearSVC")
        title = f"Precision-Recall curve |\n ws={windowsize} npeak={npeak} psteps={steps}"
        _ = display.ax_.set_title(title)
        filename = f"pr_{windowsize}_{npeak}_{steps}.png"
        path = out_dir / "pr_curve_test"
        path.mkdir(parents=True, exist_ok=True)
        filepath = path / filename
        print(filepath)
        display.figure_.savefig(filepath)
        precision_test.append(display.average_precision)


        auc_test_list.append(aucs_test)
        median_auc_test = np.median(aucs_test)
        thresh_list.append(thresh_float)
        median_auc_test_list.append(median_auc_test)
        n_samples.append(training_shape + testing_shape)

        all_y_train = np.array(all_y_train)
        all_probs_train = np.array(all_probs_train)
        fpr, tpr, thresholds = roc_curve(all_y_train, all_probs_train)
        roc_auc = auc(fpr, tpr)
        aucs_train.append(roc_auc)

        display = PrecisionRecallDisplay.from_predictions(all_y_train, all_probs_train, name="LinearSVC")
        title = f"Precision-Recall curve |\n ws={windowsize} npeak={npeak} psteps={steps}"
        _ = display.ax_.set_title(title)
        filename = f"pr_{windowsize}_{npeak}_{steps}.png"
        path = out_dir / "pr_curve_train"
        path.mkdir(parents=True, exist_ok=True)
        filepath = path / filename
        print(filepath)
        display.figure_.savefig(filepath)
        precision_train.append(display.average_precision)

        auc_train_list.append(aucs_train)
        median_auc_train = np.median(aucs_train)
        median_auc_train_list.append(median_auc_train)

    df_data = pd.DataFrame(
        {
            "auc_test_list": auc_test_list,
            "median_auc_test": median_auc_test_list,
            "auc_train_list": auc_train_list,
            "median_auc_train": median_auc_train_list,

            "thresh_list": thresh_list,
            "n_samples": n_samples,
            "p_steps_list": p_steps_list,
            "window_size_list": window_size_list,
            "n_peaks": n_peaks,
        }
    )

    print(df_data)
    fig, ax1 = plt.subplots(figsize=(10.80, 10.80))
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

    colors = list(mcolors.CSS4_COLORS.keys())
    print(colors)
    cpt = 0
    colors_ = []
    label_ = []
    for i, df in enumerate(dfs):
        dfs_ = [group for _, group in df.groupby(['window_size_list'])]
        for df_ in dfs_:
            # df_ = df_[df_['n_peaks'] <= 4]
            # if 'linear' in df_['p_steps_list'].tolist()[0]:
            #     continue
            # if 'linear' in df_['p_steps_list'].tolist()[0]:
            #     continue
            # if 'cats_LeaveOneOut_-1_-1_STD_rbf' in df_['p_steps_list'].tolist()[0]:
            #     continue
            #
            # if 'cats_LeaveOneOut_-1_-1_STD_linear' in df_['p_steps_list'].tolist()[0]:
            #     continue
            print(df_['p_steps_list'].tolist()[0])
            #
            # if len(df_["median_auc_test"]) != 4:
            #     continue
            #print(df_["n_samples"])
            label = f"Window size={df_['window_size_list'].tolist()[0]*2} sec | {'>'.join(df_['p_steps_list'].tolist()[0].split('_')[4:])}"
            ax1.plot(
                df_["n_peaks"],
                df_["median_auc_test"],
                label=label,
                marker="x",
                color=colors[cpt]
            )

            ax1.plot(
                df_["n_peaks"],
                df_["median_auc_train"],
                #label=f"Train Window size={df_['window_size_list'].tolist()[0]*2} sec | {'>'.join(df_['p_steps_list'].tolist()[0].split('_')[4:])}",
                marker=".",
                linestyle='-.',
                color=colors[cpt]
            )
            cpt +=1
            colors_.append(colors[cpt])
            label_.append(label)

    ax1.axhline(y=0.5, color='black', linestyle='--')
    fig.suptitle("Evolution of AUC(training and testing) with N peak increase")
    ax1.set_xlabel("Number of peaks")
    ax1.set_ylabel("Mean AUC")
    ax2.set_ylabel("Number of samples(high activity peak window)")
    #plt.legend()
    #ax1.legend(loc="lower right").set_visible(True)
    ax2.legend(loc="upper left").set_visible(True)

    color_data = []
    for item in colors_:
        color_data.append((item, '--'))

    ax1.legend(color_data, label_, loc="lower right",
               handler_map={tuple: AnyObjectHandler()})

    fig.tight_layout()
    filename = f"auc_per_npeak.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    filepath = out_dir / filename
    print(filepath)
    fig.savefig(filepath)


if __name__ == "__main__":
    local_run()
    # typer.run(main)

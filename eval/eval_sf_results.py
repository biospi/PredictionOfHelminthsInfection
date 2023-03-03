from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from multiprocessing import Pool

from utils.visualisation import plot_ml_report_final

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from var import marker


def mean_confidence_interval(x):
    # boot_median = [np.median(np.random.choice(x, len(x))) for _ in range(iteration)]
    x.sort()
    lo_x_boot = np.nanpercentile(x, 2.5)
    hi_x_boot = np.nanpercentile(x, 97.5)
    # print(lo_x_boot, hi_x_boot)
    return lo_x_boot, hi_x_boot


def format_cli(array):
    mean = np.nanmean(array)
    lo, hi = mean_confidence_interval(array)
    return f"{mean:.2f} ({lo:.2f}-{hi:.2f})"


def process_fold(path, i, tot):
    print(f"{i}/{tot}...")
    paths = np.array(list(path.glob("*.json")))
    if len(paths) == 0:
        return
    auc_train = []
    auc_test = []
    precision_train_0 = []
    precision_test_0 = []
    precision_train_1 = []
    precision_test_1 = []
    for filepath in paths:
        with open(filepath, "r") as fp:
            try:
                loo_result = json.load(fp)
            except Exception as e:
                return
        auc_train.append(loo_result["auc_train"])
        auc_test.append(loo_result["auc"])
        precision_train_0.append(loo_result["train_precision_score_0"])
        precision_test_0.append(loo_result["test_precision_score_0"])
        precision_train_1.append(loo_result["train_precision_score_1"])
        precision_test_1.append(loo_result["test_precision_score_1"])
    auc_train_str = format_cli(auc_train)
    auc_test_str = format_cli(auc_test)
    precision_train_0_str = format_cli(precision_train_0)
    precision_train_1_str = format_cli(precision_train_1)
    precision_test_0_str = format_cli(precision_test_0)
    precision_test_1_str = format_cli(precision_test_1)
    n_training_sample = len(loo_result["y_train"])
    n_testing_sample = len(loo_result["y_test"])
    unhealthy_label = path.parent.stem
    clf = path.parent.parent.parent.parent.stem
    if clf == "rbf":
        clf = "SVM(rbf)"
    if clf == "linear":
        clf = "SVM(linear)"
    preproc = "->".join(
        path.parent.parent.stem.split("RepeatedKFold_")[1].split("_")[3:-2]
    )
    days_a = int(path.parent.parent.stem.split("RepeatedKFold_")[1].split("_")[1])
    days_imp = int(path.parent.parent.stem.split("RepeatedKFold_")[1].split("_")[0])
    mean_auc_test = np.nanmean(auc_test)
    imputation = path.parent.parent.name
    if "gain" in imputation:
        imputation = "gain"

    if imputation != "gain":
        if "li" in imputation:
            imputation = "li"

        if "mrnn" in imputation:
            imputation = "mrnn"

    farm = path.parent.parent.name
    if "delmas" in farm.lower():
        farm = "delmas"
    if "cedara" in farm.lower():
        farm = "cedara"

    return (
        auc_test_str,
        auc_train_str,
        precision_test_0_str,
        precision_test_1_str,
        precision_train_0_str,
        precision_train_1_str,
        n_training_sample,
        n_testing_sample,
        days_a,
        days_imp,
        imputation,
        clf,
        preproc,
        unhealthy_label,
        mean_auc_test,
        path,
        auc_train,
        auc_test,
        farm,
    )


def sample_length_effect_plot(
    data_dir, df_, study_id="delmas", unhealthy="2To2", clf="lreg"
):
    df_s = df_[df_["Unhealthy"] == unhealthy]
    # df_s = df_s[df_s["Classifier"] == clf]

    df_s = df_s[df_s["Synthetic days"] == 7]
    df_s = df_s[df_s["Imputation"] == "mrnn"]
    df_s = df_s[df_s["Farm"] == study_id]
    df_s = df_s[~df_s["Pre-processing"].str.contains("CWT")]
    df_s["Pre-processing"] = df_s["Pre-processing"].fillna("None")
    df_s["Pre-processing"] = df_s["Pre-processing"] + "->" + df_s["Classifier"]
    name = "Activity days"
    metric = "mean_auc_test"
    df_s = df_s.sort_values(name)
    ax = df_s.pivot(name, "Pre-processing", metric).plot(
        figsize=(6.4, 6.4),
        kind="line",
        linestyle="--",
        rot=0,
        grid=True,
        title=f"Evolution of AUC\nwith increasing sample length",
    )
    ax.set_xticks(sorted(df_s[name].unique()))
    for n in df_s["Pre-processing"].unique():
        dat = df_s[df_s["Pre-processing"] == n].sort_values(name)
        intervals = pd.eval(dat[f"auc_test"])
        perct = np.percentile(intervals, [2.5, 50, 97.5], axis=1)
        top = perct[2, :]
        bottom = perct[0, :]
        x = dat[name].values
        ax.fill_between(x, top.astype(float), bottom.astype(float), alpha=0.1)
    a = []
    for i, line in enumerate(ax.get_lines()):
        line.set_marker(marker[i])
        a.append(line.get_label())
    ax.legend(
        ax.get_lines(),
        a,
        loc="lower left",
        title=ax.get_legend().get_title().get_text(),
    )
    ax.set(xlabel="Sample length (in days)", ylabel="AUC")
    ax.set_ylim([0.5, 1])
    filename = f"{study_id}_{metric}_{unhealthy}_{clf}_sample_length.png"
    filepath = data_dir / filename
    print(filepath)
    plt.tight_layout()
    plt.savefig(filepath)


if __name__ == "__main__":

    data_dir = Path("H:/thesis_final_march1/thesis_final_march1/main_experiment")
    folders = [x for x in data_dir.glob("*/*/*/*") if x.is_dir()]

    pool = Pool(processes=7)
    results = []
    for i, item in enumerate(folders):
        fold_data = item / "fold_data"
        if not fold_data.exists():
            continue
        # res = process_fold(fold_data)
        # results.append(res)
        results.append(pool.apply_async(process_fold, (fold_data, i, len(folders))))
    data = []
    for item in results:
        data.append(item.get())

    header = [
        "AUC test(95% CI)",
        "AUC train",
        "Class0 P-test",
        "Class1 P-test",
        "Class0 P-train",
        "Class1 P-train",
        "N train",
        "N test",
        "A-days",
        "S-days",
        "Imp",
        "Clf",
        "Pre-proc",
        "Class1",
        "mean_auc_test",
        "path",
        "auc_train",
        "auc_test",
        "Farm",
    ]
    df = pd.DataFrame(data, columns=header)
    df = df[pd.notna(df["N train"])]
    df = df[pd.notna(df["N test"])]

    df["N train"] = df["N train"].astype(int)
    df["N test"] = df["N test"].astype(int)
    df["A-days"] = df["A-days"].astype(int)
    df["S-days"] = df["S-days"].astype(int)

    df_ = df.sort_values("mean_auc_test", ascending=False)
    df_ = df_[df_["N test"] > 20]
    df_.to_csv("sf_results.csv", index=None)

    # sample_length_effect_plot(data_dir, df_, "delmas", "2To2", "lreg")

    df_ = df_.drop("mean_auc_test", axis=1)
    df_ = df_.drop("path", axis=1)
    df_ = df_.drop("auc_train", axis=1)
    df_ = df_.drop("auc_test", axis=1)
    df_2 = df_[
        [
            "AUC test(95% CI)",
            "Class0 P-test",
            "Class1 P-test",
            # "N train",
            # "N test",
            "A-days",
            "S-days",
            "Imp",
            "Clf",
            "Pre-proc",
            "Class1",
            "Farm",
        ]
    ]
    df_cwt = df_2[df_2['Pre-proc'].str.contains('CWT')]
    df_cwt_delmas = df_cwt[df_cwt["Farm"] == "delmas"]
    df_cwt_cedara = df_cwt[df_cwt["Farm"] == "cedara"]
    print(df_cwt_delmas.head(10).to_latex(index=False))
    print(df_cwt_cedara.head(10).to_latex(index=False))

    df_2 = df_2[df_2["Farm"] == "cedara"]
    df_2_h = df_2.head(10)
    print(df_2_h.to_latex(index=False))
    df_2_t = df_2.tail(10)
    print(df_2_t.to_latex(index=False))

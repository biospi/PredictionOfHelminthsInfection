from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_curve, auc
from multiprocessing import Pool, Manager
import numpy as np
import json
from tqdm import tqdm

from utils.Utils import ninefive_confidence_interval
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.legend_handler import HandlerBase


class AnyObjectHandler(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       x0, y0, width, height, fontsize, trans):
        l1 = plt.Line2D([x0,y0+width], [0.7*height,0.7*height],
                           linestyle=orig_handle[1], color=orig_handle[0])
        l2 = plt.Line2D([x0,y0+width], [0.3*height,0.3*height],
                           color=orig_handle[0])
        return [l1, l2]


def worker(
    data,
    i,
    tot,
    paths,
    ax_roc_merge,
    xaxis_train,
    xaxis_test,
    auc_list_test,
    auc_list_train

):
    all_test_y_list = []
    all_test_proba_list= []
    all_train_y_list= []
    all_train_proba_list= []
    prec_list_test= []
    prec_list_train= []

    print(f"bootstrap results progress {i}/{tot}...")
    bootstrap = np.random.choice(paths, size=len(paths), replace=True)
    # loo_pairs = []
    # for path in bootstrap:
    #     all_files_minus_one = paths.copy()
    #     all_files_minus_one = np.delete(all_files_minus_one, paths.tolist().index(path))
    #     loo_pairs.append([path, all_files_minus_one])

    all_test_proba = []
    all_test_y = []
    all_train_proba = []
    all_train_y = []
    for n, filepath in enumerate(bootstrap):
        #print(f"{n} / {len(bootstrap)}")
        # with open(filepath, 'r') as fp:
        # loo_result = json.load(fp)
        loo_result = data[filepath.stem]
        # print(len(loo_result["y_test"]))

        y_pred_proba_test = loo_result["y_pred_proba_test"]
        y_pred_proba_test = np.array(y_pred_proba_test)[:, 1]
        y_pred_proba_test = y_pred_proba_test.astype(np.float16)
        y_test = loo_result["y_test"]

        all_test_proba.extend(y_pred_proba_test)
        all_test_y.extend(y_test)
        all_test_y_list.extend(y_test)
        all_test_proba_list.extend(y_pred_proba_test)

        y_pred_proba_train = loo_result["y_pred_proba_train"]
        y_pred_proba_train = np.array(y_pred_proba_train)[:, 1]
        y_pred_proba_train = y_pred_proba_train.astype(np.float16)
        y_train = loo_result["y_train"]
        all_train_proba.extend(y_pred_proba_train)
        all_train_y.extend(y_train)
        all_train_y_list.extend(y_train)
        all_train_proba_list.extend(y_pred_proba_train)

    fpr, tpr, thresholds = roc_curve(all_test_y, all_test_proba)
    # tprs_test.append(tpr)
    # fprs_test.append(fpr)
    roc_auc = auc(fpr, tpr)
    auc_list_test.append(roc_auc)
    xaxis_test.append([fpr, tpr])
    # ax_roc_merge.plot(fpr, tpr, color="tab:blue", alpha=0.3, linewidth=1)

    fpr, tpr, thresholds = roc_curve(all_train_y, all_train_proba)
    # tprs_train.append(tpr)
    # fprs_train.append(fpr)
    roc_auc = auc(fpr, tpr)
    auc_list_train.append(roc_auc)
    xaxis_train.append([fpr, tpr])
    # ax_roc_merge.plot(fpr, tpr, color="tab:purple", alpha=0.3, linewidth=1)

    prec_list_test.append(
        precision_score(all_test_y, (np.array(all_test_proba) > 0.5).astype(int))
    )
    prec_list_train.append(
        precision_score(all_train_y, (np.array(all_train_proba) > 0.5).astype(int))
    )

    pd.DataFrame(all_test_y_list).to_pickle("all_test_y_list.pkl")
    pd.DataFrame(all_test_proba_list).to_pickle("all_test_proba_list.pkl")
    pd.DataFrame(all_train_y_list).to_pickle("all_train_y_list.pkl")
    pd.DataFrame(all_train_proba_list).to_pickle("all_train_proba_list.pkl")
    # pd.DataFrame(auc_list_test).to_pickle("auc_list_test.pkl")
    # pd.DataFrame(auc_list_train).to_pickle("auc_list_train.pkl")
    pd.DataFrame(prec_list_test).to_pickle("prec_list_test.pkl")
    pd.DataFrame(prec_list_train).to_pickle("prec_list_train.pkl")

    print(f"{i}/{tot} done.")


def main(path=None, n_bootstrap=100, n_job=6):
    print("loading data...")
    paths = np.array(list(path.glob("*.json")))
    if len(paths) == 0:
        return
    data = {}
    fig_roc_merge, ax_roc_merge = plt.subplots(figsize=(6, 6))
    for filepath in tqdm(paths):
        with open(filepath, "r") as fp:
            try:
                loo_result = json.load(fp)
            except Exception as e:
                return

            training_size = loo_result["training_shape"][0]
            testing_size = loo_result["testing_shape"][0]
            n_peaks = int(filepath.parent.parent.parent.parent.stem.split("__")[1])
            window_size = int(loo_result["training_shape"][1] / 60)
            clf = filepath.parent.parent.parent.stem[23:].split("_")[-1]
            if clf == "rbf":
                clf = "SVM(rbf)"
            if clf == "linear":
                clf = "SVM(linear)"
            pre_proc = "->".join(
                filepath.parent.parent.parent.stem[23:].split("_")[:-1]
            )

            data[filepath.stem] = {
                "y_pred_proba_test": loo_result["y_pred_proba_test"],
                "y_test": loo_result["y_test"],
                "y_pred_proba_train": loo_result["y_pred_proba_train"],
                "y_train": loo_result["y_train"],
                "training_size": training_size,
                "testing_size": testing_size,
                "n_peaks": n_peaks,
                "window_size": window_size,
                "clf": clf,
                "pre_proc": pre_proc,
            }

    print("start bootstrap...")
    pool = Pool(processes=n_job)
    with Manager() as manager:
        auc_list_test = manager.list()
        auc_list_train = manager.list()
        prec_list_test = manager.list()
        prec_list_train = manager.list()
        # tprs_test = manager.list()
        # tprs_train = manager.list()
        # fprs_test = manager.list()
        # fprs_train = manager.list()
        xaxis_train = manager.list()
        xaxis_test = manager.list()
        all_test_y_list = manager.list()
        all_test_proba_list = manager.list()
        all_train_y_list = manager.list()
        all_train_proba_list = manager.list()
        for i in range(n_bootstrap):
            pool.apply_async(
                worker,
                (
                    data,
                    i,
                    n_bootstrap,
                    paths,
                    ax_roc_merge,
                    xaxis_train,
                    xaxis_test,
                    auc_list_test,
                    auc_list_train
                ),
            )
        pool.close()
        pool.join()
        pool.terminate()
        print("pool done.")
        xaxis_train = list(xaxis_train)
        xaxis_test = list(xaxis_test)
        auc_list_test = list(auc_list_test)
        auc_list_train = list(auc_list_train)

    all_test_y_list = pd.read_pickle("all_test_y_list.pkl").values
    all_test_proba_list = pd.read_pickle("all_test_proba_list.pkl").values
    all_train_y_list = pd.read_pickle("all_train_y_list.pkl").values
    all_train_proba_list = pd.read_pickle("all_train_proba_list.pkl").values
    # auc_list_test = pd.read_pickle("auc_list_test.pkl").values
    # auc_list_train = pd.read_pickle("auc_list_train.pkl").values
    prec_list_test = pd.read_pickle("prec_list_test.pkl").values
    prec_list_train = pd.read_pickle("prec_list_train.pkl").values

    print("building roc...")
    median_auc_test = np.median(auc_list_test)
    lo_test_auc, hi_test_auc = ninefive_confidence_interval(auc_list_test)
    print(
        f"median Test AUC = {median_auc_test:.2f}, 95% CI [{lo_test_auc:.2f}, {hi_test_auc:.2f}] )"
    )

    median_auc_train = np.median(auc_list_train)
    lo_train_auc, hi_train_auc = ninefive_confidence_interval(auc_list_train)
    print(
        f"median Train AUC = {median_auc_train:.2f}, 95% CI [{lo_train_auc:.2f}, {hi_train_auc:.2f}] )"
    )

    median_prec_test = np.median(prec_list_test)
    lo_test_prec, hi_test_prec = ninefive_confidence_interval(prec_list_test)
    print(
        f"median Test prec = {median_prec_test:.2f}, 95% CI [{lo_test_prec:.2f}, {hi_test_prec:.2f}] )"
    )

    median_prec_train = np.median(prec_list_train)
    lo_train_prec, hi_train_prec = ninefive_confidence_interval(prec_list_train)
    print(
        f"median Train prec = {median_prec_train:.2f}, 95% CI [{lo_train_prec:.2f}, {hi_train_prec:.2f}] )"
    )

    for fpr, tpr in xaxis_train:
        ax_roc_merge.plot(fpr, tpr, color="tab:purple", alpha=0.3, linewidth=1)

    for fpr, tpr in xaxis_test:
        ax_roc_merge.plot(fpr, tpr, color="tab:blue", alpha=0.3, linewidth=1)

    ax_roc_merge.plot(
        [0, 1], [0, 1], linestyle="--", lw=2, color="orange", label="Chance", alpha=1
    )

    # mean_fpr_test = np.mean(fprs_test, axis=0)
    # mean_tpr_test = np.mean(tprs_test, axis=0)
    mean_fpr_test, mean_tpr_test, thresholds = roc_curve(
        all_test_y_list, all_test_proba_list
    )
    label = f"Mean ROC Test (Median AUC = {median_auc_test:.2f}, 95% CI [{lo_test_auc:.4f}, {hi_test_auc:.4f}] )"
    ax_roc_merge.plot(
        mean_fpr_test, mean_tpr_test, color="black", label=label, lw=2, alpha=1
    )
    ax_roc_merge.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title=f"(Training/Testing data) Receiver operating characteristic",
    )
    ax_roc_merge.set_xlabel("False positive rate")
    ax_roc_merge.set_ylabel("True positive rate")
    ax_roc_merge.legend(loc="lower right")
    # fig.show()

    # mean_tpr_train = np.mean(tprs_train, axis=0)
    # mean_fpr_train = np.mean(fprs_train, axis=0)
    # mean_tpr_train[-1] = 1.0
    mean_fpr_train, mean_tpr_train, thresholds = roc_curve(
        all_train_y_list, all_train_proba_list
    )
    label = f"Mean ROC Training (Median AUC = {median_auc_train:.2f}, 95% CI [{lo_train_auc:.4f}, {hi_train_auc:.4f}] )"

    ax_roc_merge.plot(
        mean_fpr_train, mean_tpr_train, color="red", label=label, lw=2, alpha=1
    )
    ax_roc_merge.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title=f"(Training data) Receiver operating characteristic",
    )
    ax_roc_merge.legend(loc="lower right")

    fig_roc_merge.tight_layout()
    path = path.parent.parent.parent / "roc_curve"
    path.mkdir(parents=True, exist_ok=True)
    # final_path = path / f"{tag}_roc_{classifier_name}.png"
    # print(final_path)
    # fig.savefig(final_path)

    final_path = path / f"{path.parent.stem}.png"
    print(final_path)
    fig_roc_merge.savefig(final_path)

    return [
        f"{median_auc_test:.2f} ({lo_test_auc:.2f}-{hi_test_auc:.2f})",
        f"{median_auc_train:.2f} ({lo_train_auc:.2f}-{hi_train_auc:.2f})",
        f"{median_prec_test:.2f} ({lo_test_prec:.2f}-{hi_test_prec:.2f})",
        f"{median_prec_train:.2f} ({lo_train_prec:.2f}-{hi_train_prec:.2f})",
        training_size,
        testing_size,
        n_peaks,
        window_size,
        clf,
        pre_proc,
        median_auc_test,
        paths,
    ]


if __name__ == "__main__":
    out_dir = Path("E:/Cats/article/ml_build_permutations_qnf_final/")
    results = []
    folders = [
        x
        for x in out_dir.glob("*/*/*")
        if x.is_dir()
    ]
    for i, item in enumerate(folders):
        # if "1000__005__0_00100__030\cats_LeaveOneOut_-1_-1_QN_rbf" not in str(item):
        #     continue
        print(item)
        print(f"{i}/{len(folders)}...")
        res = main(Path(f"{item}/fold_data"))
        if res is not None:
            results.append(res)

    df = pd.DataFrame(
        results,
        columns=[
            "AUC testing (95% CI)",
            "AUC training (95% CI)",
            "Class1 Precision testing (95% CI)",
            "Class1 Precision training (95% CI)",
            "N training samples",
            "N testing samples",
            "N peaks",
            "Sample length (minutes)",
            "Classifier",
            "Pre-processing",
            "median_auc_test",
            "path",
        ],
    )
    # df_ = df.sort_values("median_auc_test", ascending=False)
    # df_ = df_.drop("median_auc_test", axis=1)
    # df_ = df_.drop("path", axis=1)
    # df_ = df_.head(20)
    # print(df_.to_latex(index=False))
    # df.to_csv("cat_result_table.csv", index=False)

    print(df)
    fig, ax1 = plt.subplots(figsize=(10.80, 10.80))
    ax2 = ax1.twinx()
    dfs = [group for _, group in df.groupby(['p_steps_list'])]

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
    filename = f"auc_per_npeak_bootstrap.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    filepath = out_dir / filename
    print(filepath)
    fig.savefig(filepath)
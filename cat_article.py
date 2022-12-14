import time
from multiprocessing import Manager, Pool
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from model.data_loader import load_activity_data
from model.svm import cross_validate_svm_fast
from utils._custom_split import LeaveNOut
from utils._normalisation import QuotientNormalizer
from pathlib import Path
from sys import exit

from utils.visualisation import plot_umap

n_job = 7


def fold_worker(
    kernel, i, tot, fold, all_y_test, all_y_train, all_y_pred_test, all_y_pred_train
):
    X_train = fold[1].iloc[:, :-1]
    y_train = fold[1]["label"]
    X_test = fold[0].iloc[:, :-1]
    y_test = fold[0]["label"]
    clf = SVC(kernel=kernel, probability=True)
    clf.fit(X_train, y_train)
    y_pred_test = clf.decision_function(X_test)
    y_pred_train = clf.decision_function(X_train)
    all_y_test.extend(y_test)
    all_y_train.extend(y_train)
    all_y_pred_test.extend(y_pred_test)
    all_y_pred_train.extend(y_pred_train)
    print(f"FOLD {i}/{tot}, cat {fold[2]}, kernel {kernel}:")
    print(f"y_test      --> {y_test.values}")
    print(f"y_pred_test --> {y_pred_test}")
    # print(f"y_pred_train --> {y_pred_train}")
    # print(f"y_train      --> \n{y_train.values}")
    # print(f"y_pred_train -->")
    # print(f"{y_pred_train}")
    print(
        f"training y balance --> {y_train.value_counts().to_dict()} 1: {y_train.value_counts().to_dict()[1] / np.sum(list(y_train.value_counts().to_dict().values())):.2f}% 0: {y_train.value_counts().to_dict()[0] / np.sum(list(y_train.value_counts().to_dict().values())):.2f}%"
    )
    print(f"testing y balance --> {y_test.value_counts().to_dict()}")
    print("**************************************************************")


meta_columns = [
    "label",
    "id",
    "imputed_days",
    "date",
    "health",
    "target",
    "age",
    "name",
    "mobility_score",
]

if __name__ == "__main__":
    output_dir = Path("E:/Cats/debug2")
    path = "E:/Cats/build_permutations_debug/no_windows_mins/dataset/training_sets/full/activity.csv"
    N_META = 9
    df = pd.read_csv(path, header=None)
    header = list(df.columns.values)
    header[-9] = "label"
    header[-8] = "id"
    header[-7] = "imputed_days"
    header[-6] = "date"
    header[-5] = "health"
    header[-4] = "target"
    header[-3] = "age"
    header[-2] = "name"
    header[-1] = "mobility_score"

    df.columns = header

    df["target"] = df["health"]
    # individual_to_ignore = ["'MrDudley'", "'Oliver_F'", "'Lucy'"]
    # data_frame = df.loc[~df['name'].isin(individual_to_ignore)]
    df["date"] = [x.replace("'", '') for x in df['date']]

    df = df[df.nunique(1) > 10]
    df = df.dropna(subset=df.columns[: -N_META], how="all")
    df.iloc[:, : -N_META] = df.iloc[:, : -N_META].clip(lower=0)
    df.iloc[:, :-N_META] = QuotientNormalizer(
        out_dir=output_dir, output_graph=True, enable_qn_peak_filter=False
    ).transform(df.iloc[:, :-N_META].values)

    #df.iloc[:, :-N_META] = StandardScaler(with_mean=True, with_std=True).fit_transform(df.iloc[:, :-N_META].values)

    folds = []
    # for cat in df["id"].unique():
    #     df_test = df[df["id"] == cat].iloc[:, :-8]
    #     df_train = df[df["id"] != cat].iloc[:, :-8]
    #     folds.append([df_test, df_train, df[df["id"] == cat]["name"].values[0]])
    #
    # for kernel in ["linear"]:
    #     print("-------------------------------------------------------------------------------------------")
    #     print(f"                                       {kernel}                                           ")
    #     print("-------------------------------------------------------------------------------------------")
    #     with Manager() as manager:
    #         pool = Pool(processes=n_job)
    #         all_y_test = manager.list()
    #         all_y_train = manager.list()
    #         all_y_pred_test = manager.list()
    #         all_y_pred_train = manager.list()
    #         start = time.time()
    #         for i, fold in enumerate(folds):
    #             pool.apply_async(
    #                 fold_worker,
    #                 (
    #                     kernel,
    #                     i,
    #                     len(folds),
    #                     fold,
    #                     all_y_test,
    #                     all_y_train,
    #                     all_y_pred_test,
    #                     all_y_pred_train,
    #                 ),
    #             )
    #         pool.close()
    #         pool.join()
    #
    #         all_y_test = list(all_y_test)
    #         all_y_train = list(all_y_train)
    #         all_y_pred_test = list(all_y_pred_test)
    #         all_y_pred_train = list(all_y_pred_train)
    #         end = time.time()
    #         print("total time (s)= " + str(end - start))
    #
    #         all_y_test = np.array(all_y_test)
    #         all_y_pred_test = np.array(all_y_pred_test)
    #         # if len(all_y_pred_test.shape) > 1:
    #         #     all_y_pred_test = all_y_pred_test[:, 1]
    #
    #         all_y_pred_train = np.array(all_y_pred_train)
    #         # if len(all_y_pred_train.shape) > 1:
    #         #     all_y_pred_train = all_y_pred_train[:, 1]
    #
    #         fpr, tpr, _ = roc_curve(all_y_test, all_y_pred_test)
    #         roc_auc = auc(fpr, tpr)
    #         # print(all_y_test)
    #         # print(all_y_pred_test)
    #         print(f"AUC TEST={roc_auc}")
    #         fpr, tpr, _ = roc_curve(all_y_train, all_y_pred_train)
    #         roc_auc = auc(fpr, tpr)
    #         # print(all_y_train)
    #         # print(all_y_pred_train)
    #         print(f"AUC TRAIN={roc_auc}")
    #
    #         plt.plot(all_y_train[0:1000])
    #         plt.plot(all_y_pred_train[0:1000])
    #         plt.show()

    #exit()
###############################################################################################################################

    individual_to_ignore = ["MrDudley", "Oliver_F", "Lucy"]
    individual_to_ignore = []

    (data_frame, meta_data, meta_data_short, _, _, label_series, samples, seasons_features) = load_activity_data(
        output_dir,
        meta_columns,
        path,
        -1,
        ["0.0"],
        ["0.1"],
        imputed_days=-1,
        preprocessing_steps=[],
        meta_cols_str=["health", "label", "id"],
        sampling='S',
        individual_to_keep=[],
        individual_to_ignore=individual_to_ignore,
        resolution=None
    )

    #df.iloc[:, : -N_META] = df.iloc[:, : -N_META].clip(lower=0)
    #df.iloc[:, : -N_META] = df.iloc[:, : -N_META].astype(float).interpolate(axis=1, limit_direction="both")
    data_frame["health"] = data_frame["target"]
    data_frame.iloc[:, :-N_META] = QuotientNormalizer(
        out_dir=output_dir, output_graph=True, enable_qn_peak_filter=True
    ).transform(data_frame.iloc[:, :-N_META].values)

    data_frame.iloc[:, :-N_META] = StandardScaler(with_mean=True, with_std=True).fit_transform(data_frame.iloc[:, :-N_META].values)

    sample_idxs = data_frame.index.tolist()
    animal_ids = data_frame["id"].astype(str).tolist()
    cross_validation_method = LeaveNOut(
        animal_ids, sample_idxs, stratified=False, verbose=True, leaven=1, individual_to_test=[]
    )

    y_col = "target"
    ids = data_frame["id"].values
    y_h = data_frame["health"].values.flatten()
    y_h = y_h.astype(int)
    y = data_frame[y_col].values.flatten()
    y = y.astype(int)

    # remove meta columns
    print("creating X...")
    X = data_frame.iloc[:, np.array([str(x).isnumeric() or x in [] for x in data_frame.columns])]
    X.columns = list(range(X.shape[1]))
    X = X.values

    sample_dates = pd.to_datetime(
        data_frame["date"], format="%d/%m/%Y"
    ).values.astype(float)

    scores, scores_proba = cross_validate_svm_fast(
        False,
        ["linear"],
        output_dir,
        ["QN"],
        "LeaveOneOut",
        -1,
        label_series,
        cross_validation_method,
        X,
        y,
        y_h,
        ids,
        meta_data,
        meta_columns,
        meta_data_short,
        sample_dates,
        False,
        n_job,
        False,
        False
    )

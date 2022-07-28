import typer
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
import glob
import seaborn as sns
from sys import exit
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from model.data_loader import load_activity_data
from model.svm import process_ml
from preprocessing.preprocessing import apply_preprocessing_steps
from utils.visualisation import plot_umap, plot_time_pca, plot_time_pls
from natsort import natsorted


def interpolate_time(a, new_length):
    old_indices = np.arange(0, len(a))
    new_indices = np.linspace(0, len(a) - 1, new_length)
    spl = UnivariateSpline(old_indices, a, k=3, s=0)
    new_array = spl(new_indices)
    new_array[0] = 0
    return new_array


def build_samples(df_, seq, label, health, target):
    #print(df_)
    #df = df_.iloc[:, np.array([str(x).isnumeric() for x in df_.columns])]
    n = len(seq)
    dfs_ = [df_[i:i + n] for i in range(0, df_.shape[0], n)]
    samples = []
    for item in dfs_:
        item['health'] = health
        item['target'] = target
        item['label'] = label
        item = item.drop('diff', 1)
        item = item.drop('datetime', 1)
        meta_col = item.columns[[not str(x).isnumeric() for x in item.columns]]
        n_meta = len(meta_col)
        meta = item[meta_col]
        meta_ = meta.iloc[0, :]
        meta_["imputed_days"] = sum(meta["imputed_days"])
        sample_a = np.concatenate(item.iloc[:, :-n_meta].values)
        sample = sample_a.tolist() + meta_.tolist()
        sample = np.array(sample)
        samples.append(sample)

    return samples


def plot_ribbon(path, data, title, y_label, days):
    df = pd.DataFrame.from_dict(data, orient="index")
    print(df)
    time = []
    acc = []
    for index, row in df.iterrows():
        print(row[0], row[1])
        for n in range(df.shape[1]):
            time.append(index)
            acc.append(row[n])
    data_dict = {"time": time, "acc": acc}
    df = pd.DataFrame.from_dict(data_dict)
    print(df)
    time_axis = interpolate_time(np.arange(days + 1), len(df["time"]))
    time_axis = time_axis.tolist()
    time_axis_s = []
    for t in time_axis:
        time_axis_s.append("%d" % t)

    fig, ax = plt.subplots(figsize=(15, 5))
    sns.lineplot(x=df["time"], y="acc", data=df, marker="o", ax=ax)
    fig.suptitle(title)
    # ax = df.copy().plot.box(grid=True, patch_artist=True, title=title, figsize=(10, 7))
    ax.set_xlabel("days")
    ax.set_ylabel(y_label)

    labels = [item.get_text() for item in ax.get_xticklabels()]
    m_d = max(df["time"].to_list()) + 1
    labels_ = interpolate_time(np.arange(days + 1), m_d)
    l = []
    for i, item in enumerate(labels_):
        l.append("%.1f" % float(item))

    # labels = ['0'] + labels + ['0']
    # ax.set_xticklabels(labels)
    ticks = list(range(m_d))
    ax.set_xticks(ticks)
    ax.set_xticklabels(l)

    print("labels", labels)

    # ax.set_xticklabels(time_axis_s)
    fig.tight_layout()
    file_path = path / "model_auc_progression.png"
    print(file_path)
    fig.savefig(str(file_path))


def plot_progression(output_dir, days, window, famacha_healthy, famacha_unhealthy, shape_healthy, shape_unhealthy):
    print("plot progression...")
    files = [x for x in list(output_dir.glob("**/*.csv")) if 'classification_report_days' in str(x)]
    files = natsorted(files)
    aucs = {}
    for i, file in enumerate(files):
        print(file)
        df = pd.read_csv(str(file), converters={'roc_auc_scores': eval})
        a = df["roc_auc_scores"][0]
        aucs[i] = a

    plot_ribbon(
        output_dir,
        aucs,
        f"Classifier Auc over time during increase of the FAMACHA score (window={window})\nhealthy:{famacha_healthy} {shape_healthy} unhealthy:{famacha_unhealthy} {shape_unhealthy}",
        "Auc",
        days,
    )


def main_(
    output_dir: Path = typer.Option(
        ..., exists=False, file_okay=False, dir_okay=True, resolve_path=True
    ),
    dataset_file: Path = typer.Option(
        ..., exists=True, file_okay=True, dir_okay=False, resolve_path=True
    ),
    class_healthy_label: List[str] = ["1To1"],
    class_unhealthy_label: List[str] = ["2To2"],
    famacha_healthy: List[str] = ["1To1", "1To1"],
    famacha_unhealthy: List[str] = ["2To2", "2To2"],
    preprocessing_steps: List[str] = ["QN", "ANSCOMBE", "LOG"],
    n_activity_days: int = 7,
    n_imputed_days: int = 7,
    meta_columns: List[str] = [
        "label",
        "id",
        "imputed_days",
        "date",
        "health",
        "target",
    ],
    n_scales: int = 9,
    sub_sample_scales: int = 1,
    n_splits: int = 5,
    n_repeats: int = 10,
    cv: str = "RepeatedKFold",
    wavelet_f0: int = 6,
    sfft_window: int = 60,
    add_feature: List[str] = [],
    meta_col_str: List[str] = ["health", "label", "date"],
    svc_kernel: List[str] = ["linear"],
    study_id: str = "study",
    sampling: str = "T",
    output_qn_graph: bool = True,
    add_seasons_to_features: bool = False,
    enable_downsample_df: bool = False,
    window: int = 1440*3,
    stride: int = 1440,
    days_between: int = 7,
    back_to_back: bool = True,
    n_job: int = 7,
):
    """This script builds...\n
    Args:\n
    """

    print(f"loading dataset file {dataset_file} ...")
    (
        data_frame,
        meta_data,
        meta_data_short,
        _,
        _,
        label_series,
        samples,
        _
    ) = load_activity_data(
        output_dir,
        meta_columns,
        dataset_file,
        n_activity_days,
        class_healthy_label,
        class_unhealthy_label,
        imputed_days=n_imputed_days,
        preprocessing_steps=preprocessing_steps,
        meta_cols_str=meta_col_str
    )
    #print(data_frame)
    sample_dates = pd.to_datetime(
        data_frame["date"], format="%d/%m/%Y"
    ).values.astype(float)
    animal_ids = data_frame["id"].astype(str).values

    step_slug = "_".join(preprocessing_steps)

    df_processed, _, _ = apply_preprocessing_steps(
        meta_columns,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        data_frame.copy(),
        output_dir,
        preprocessing_steps,
        class_healthy_label,
        class_unhealthy_label,
        clf_name="SVM",
        n_scales=None,
        farm_name=study_id,
        keep_meta=False,
    )

    df_processed["target"] = pd.to_numeric(df_processed["target"])
    df_processed["health"] = pd.to_numeric(df_processed["health"])

    shape_healthy = df_processed[df_processed["health"] == 0].shape
    shape_unhealthy = df_processed[df_processed["health"] == 1].shape

    df_target = df_processed[["target", "health"]]
    df_activity_window = df_processed.iloc[:, np.array([str(x).isnumeric() for x in df_processed.columns])]
    cpt = 0
    for i in range(0, df_activity_window.shape[1] - window, stride):
        start = i
        end = start + window
        print(start, end)
        df_a_w = df_activity_window.iloc[:, start:end]
        df_week = pd.concat([df_a_w, df_target], axis=1)
        print(df_week)
        process_ml(
            svc_kernel,
            add_feature,
            animal_ids,#meta
            animal_ids,#meta
            output_dir / f"week_{str(cpt).zfill(3)}",
            animal_ids,
            sample_dates,
            df_week,
            n_activity_days,
            n_imputed_days,
            study_id,
            step_slug,
            n_splits,
            n_repeats,
            sampling,
            enable_downsample_df,
            label_series,
            class_healthy_label,
            class_unhealthy_label,
            meta_columns,
            add_seasons_to_features,
            cv=cv,
            n_job=n_job,
        )
        cpt += 1

    plot_progression(output_dir, days_between, window, class_healthy_label, class_unhealthy_label, shape_healthy, shape_unhealthy)


def main(
    output_dir: Path = typer.Option(
        ..., exists=False, file_okay=False, dir_okay=True, resolve_path=True
    ),
    dataset_file: Path = typer.Option(
        ..., exists=True, file_okay=True, dir_okay=False, resolve_path=True
    ),
    class_healthy_label: List[str] = ["1To1"],
    class_unhealthy_label: List[str] = ["2To2"],
    famacha_healthy: List[str] = ["1To1", "1To1"],
    famacha_unhealthy: List[str] = ["2To2", "2To2"],
    preprocessing_steps: List[str] = ["QN", "ANSCOMBE", "LOG"],
    n_activity_days: int = 7,
    n_imputed_days: int = 7,
    meta_columns: List[str] = [
        "label",
        "id",
        "imputed_days",
        "date",
        "health",
        "target",
    ],
    n_scales: int = 9,
    sub_sample_scales: int = 1,
    n_splits: int = 5,
    n_repeats: int = 10,
    cv: str = "RepeatedKFold",
    wavelet_f0: int = 6,
    sfft_window: int = 60,
    add_feature: List[str] = [],
    meta_col_str: List[str] = ["health", "label", "date"],
    svc_kernel: List[str] = ["linear"],
    study_id: str = "study",
    sampling: str = "T",
    output_qn_graph: bool = True,
    add_seasons_to_features: bool = False,
    enable_downsample_df: bool = False,
    window: int = 1440*2,
    stride: int = 1440,
    days_between: int = 7,
    back_to_back: bool = True,
    n_aug: int = 5,
    n_job: int = 7,
):
    """This script builds...\n
    Args:\n
    """

    print(f"loading dataset file {dataset_file} ...")
    (
        data_frame,
        meta_data,
        meta_data_short,
        _,
        _,
        label_series,
        samples,
        _
    ) = load_activity_data(
        output_dir,
        meta_columns,
        dataset_file,
        n_activity_days,
        class_healthy_label,
        class_unhealthy_label,
        imputed_days=n_imputed_days,
        preprocessing_steps=preprocessing_steps,
        meta_cols_str=meta_col_str
    )
    #print(data_frame)
    data_frame["datetime"] = pd.to_datetime(data_frame['date'], format="%d/%m/%Y")
    data_frame = data_frame.sort_values(by='datetime')
    dfs = [g for _, g in data_frame.groupby(['id'])]

    label_series_inverse = dict((v, k) for k, v in label_series.items())

    unhealthy_samples = []
    for df in dfs:
        df["diff"] = df["datetime"].diff() / np.timedelta64(1, 'D')
        if back_to_back:
            df = df[df["diff"] == days_between]
        df = df.reset_index(drop=True)
        a = df["label"].tolist()
        b = famacha_unhealthy
        idxs = [list(range(i, i+len(b))) for i in range(len(a)) if a[i:i+len(b)] == b]
        idxs = np.array(idxs).flatten()
        df_f = df.loc[idxs]
        for item in build_samples(df_f, b, '2To2', 1, label_series_inverse['2To2']):
            unhealthy_samples.append(item)

    healthy_samples = []
    for df in dfs:
        df["diff"] = df["datetime"].diff() / np.timedelta64(1, 'D')
        if back_to_back:
            df = df[df["diff"] == days_between]
        df = df.reset_index(drop=True)
        a = df["label"].tolist()
        b = famacha_healthy
        idxs = [list(range(i, i+len(b))) for i in range(len(a)) if a[i:i+len(b)] == b]
        idxs = np.array(idxs).flatten()
        df_f = df.loc[idxs]
        for item in build_samples(df_f, b, '1To1', 0, label_series_inverse['1To1']):
            healthy_samples.append(item)

    data_frame = pd.DataFrame(np.concatenate([unhealthy_samples, healthy_samples]))
    header = list(data_frame.columns)
    for i, meta in enumerate(meta_columns[::-1]):
        header[-i-1] = meta
    data_frame.columns = header

    sample_dates = pd.to_datetime(
        data_frame["date"], format="%d/%m/%Y"
    ).values.astype(float)
    animal_ids = data_frame["id"].astype(str).values

    step_slug = "_".join(preprocessing_steps)
    df_processed, df_processed_meta = apply_preprocessing_steps(
        meta_columns,
        n_activity_days,
        None,
        None,
        sfft_window,
        wavelet_f0,
        animal_ids,
        data_frame.copy(),
        output_dir,
        preprocessing_steps,
        class_healthy_label,
        class_unhealthy_label,
        clf_name="SVM",
        output_dim=data_frame.shape[0],
        n_scales=n_scales,
        sub_sample_scales=sub_sample_scales,
        output_qn_graph=output_qn_graph
    )

    plot_umap(
        meta_columns,
        df_processed_meta.copy(),
        output_dir / f"umap_{step_slug}",
        label_series,
        title=f"UMAP after {step_slug}",
    )

    plot_time_pca(
        len(meta_columns),
        df_processed_meta,
        output_dir / f"pca_{step_slug}",
        label_series,
        title=f"PCA after {step_slug}",
    )

    plot_time_pls(
        len(meta_columns),
        df_processed_meta.copy(),
        output_dir / f"pls_{step_slug}",
        label_series,
        title=f"PLS after {step_slug}",
    )

    df_processed["target"] = pd.to_numeric(df_processed["target"])
    df_processed["health"] = pd.to_numeric(df_processed["health"])

    #augment data here
    # if n_aug > 0:
    #     df_processed, animal_ids, sample_dates,  _, _ = augment(df_processed, n_aug, animal_ids, meta_data, meta_data_short, sample_dates)

    shape_healthy = df_processed[df_processed["health"] == 0].shape
    shape_unhealthy = df_processed[df_processed["health"] == 1].shape

    df_target = df_processed[["target", "health"]]
    df_activity_window = df_processed.iloc[:, np.array([str(x).isnumeric() for x in df_processed.columns])]
    cpt = 0
    for i in range(0, df_activity_window.shape[1] - window, stride):
        start = i
        end = start + window
        print(start, end)
        df_a_w = df_activity_window.iloc[:, start:end]
        df_week = pd.concat([df_a_w, df_target], axis=1)
        print(df_week)
        process_ml(
            svc_kernel,
            add_feature,
            animal_ids,#meta
            animal_ids,#meta
            output_dir / f"week_{str(cpt).zfill(3)}",
            animal_ids,
            sample_dates,
            df_week,
            n_activity_days,
            n_imputed_days,
            study_id,
            step_slug,
            n_splits,
            n_repeats,
            sampling,
            enable_downsample_df,
            label_series,
            class_healthy_label,
            class_unhealthy_label,
            meta_columns,
            add_seasons_to_features,
            cv=cv,
            augment_training=n_aug,
            n_job=n_job,
        )
        cpt += 1

    plot_progression(output_dir, len(famacha_healthy) * days_between, window, famacha_healthy, famacha_unhealthy, shape_healthy, shape_unhealthy)


if __name__ == "__main__":
    #typer.run(main)
    # main(Path(f'E:/Data2/debug/test_distance_validation_0'),
    #      Path("E:/Data2/debug3/delmas/dataset4_mrnn_7day/activity_farmid_dbft_7_1min.csv"),
    #      famacha_healthy=["1To1", "1To1"], famacha_unhealthy=["2To2", "2To2"], window=1440)
    # main(Path(f'E:/Data2/debug2/test_distance_validation_debug10'),
    #      Path("E:/Data2/debug3/delmas/dataset4_mrnn_7day/activity_farmid_dbft_7_1min.csv"),
    #      famacha_healthy=["1To1", "1To1"], famacha_unhealthy=["1To2", "2To2"], back_to_back=True, n_aug=10)

    for w in [1440]:
        main_(Path(f'E:/thesis/test_distance_validation_debug_w_{w}'),
             Path("E:/thesis/datasets/delmas/datasetmrnn21_17/activity_farmid_dbft_21_1min.csv"),
             famacha_healthy=["1To1"], famacha_unhealthy=["2To2"], back_to_back=True,
             study_id="delmas", window=w, stride=1440, n_activity_days=21)

    # for w in [1440*5, 1440*3, 1440]:
    #     for a in [15, 20]:
    #         main(Path(f'E:/Data2/debug2/test_distance_validation_debug_w_{w}_a_{a}'),
    #              Path("E:/Data2/debug3/delmas/dataset4_mrnn_7day/activity_farmid_dbft_7_1min.csv"),
    #              famacha_healthy=["1To1", "1To1"], famacha_unhealthy=["1To2", "2To2"], back_to_back=False, n_aug=a,
    #              study_id="delmas", window=w)

    # main(Path(f'E:/Data2/debug2/test_distance_validation_1'),
    #      Path("E:/Data2/debug3/delmas/dataset4_mrnn_7day/activity_farmid_dbft_7_1min.csv"),
    #      famacha_healthy=["1To1"], famacha_unhealthy=["2To2"], window=1440, back_to_back=False)
    #
    # main(Path(f'E:/Data2/debug2/test_distance_validation_2'),
    #      Path("E:/Data2/debug3/delmas/dataset4_mrnn_7day/activity_farmid_dbft_7_1min.csv"),
    #      famacha_healthy=["1To1", "1To1"], famacha_unhealthy=["2To2", "2To2"], window=1440, back_to_back=False)
    #
    # main(Path(f'E:/Data2/debug2/test_distance_validation_3'),
    #      Path("E:/Data2/debug3/delmas/dataset4_mrnn_7day/activity_farmid_dbft_7_1min.csv"),
    #      famacha_healthy=["1To1", "1To1"], famacha_unhealthy=["1To2", "2To2"], window=1440, back_to_back=False)
    #
    # main(Path(f'E:/Data2/debug2/test_distance_validation_4'),
    #      Path("E:/Data2/debug3/delmas/dataset4_mrnn_7day/activity_farmid_dbft_7_1min.csv"),
    #      famacha_healthy=["1To1"], famacha_unhealthy=["2To2"], window=1440, back_to_back=True)
    #
    # main(Path(f'E:/Data2/debug2/test_distance_validation_5'),
    #      Path("E:/Data2/debug3/delmas/dataset4_mrnn_7day/activity_farmid_dbft_7_1min.csv"),
    #      famacha_healthy=["1To1", "1To1"], famacha_unhealthy=["2To2", "2To2"], window=1440, back_to_back=True)
    #
    # main(Path(f'E:/Data2/debug2/test_distance_validation_6'),
    #      Path("E:/Data2/debug3/delmas/dataset4_mrnn_7day/activity_farmid_dbft_7_1min.csv"),
    #      famacha_healthy=["1To1", "1To1"], famacha_unhealthy=["1To2", "2To2"], window=1440, back_to_back=True)
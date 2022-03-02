import typer
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np

from model.data_loader import load_activity_data
from model.svm import process_data_frame_svm
from preprocessing.preprocessing import apply_preprocessing_steps
from utils.visualisation import plot_umap, plot_time_pca, plot_time_pls


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


def main(
    output_dir: Path = typer.Option(
        ..., exists=False, file_okay=False, dir_okay=True, resolve_path=True
    ),
    dataset_file: Path = typer.Option(
        ..., exists=True, file_okay=True, dir_okay=False, resolve_path=True
    ),
    class_healthy_label: List[str] = ["1To1"],
    class_unhealthy_label: List[str] = ["2To2"],
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
    cv: str = "RepeatedStratifiedKFold",
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
    stride: int = 360,
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
        df = df[df["diff"] == 7]
        df = df.reset_index(drop=True)
        a = df["label"].tolist()
        b = ["2To2"]
        idxs = [list(range(i, i+len(b))) for i in range(len(a)) if a[i:i+len(b)] == b]
        idxs = np.array(idxs).flatten()
        df_f = df.loc[idxs]
        for item in build_samples(df_f, b, '2To2', 1, label_series_inverse['2To2']):
            unhealthy_samples.append(item)

    healthy_samples = []
    for df in dfs:
        df["diff"] = df["datetime"].diff() / np.timedelta64(1, 'D')
        df = df[df["diff"] == 7]
        df = df.reset_index(drop=True)
        a = df["label"].tolist()
        b = ["1To1"]
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
    animal_ids = data_frame["id"].astype(str).tolist()

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

    df_target = df_processed[["target", "health"]]
    df_activity_window = df_processed.iloc[:, np.array([str(x).isnumeric() for x in df_processed.columns])]
    cpt = 0
    for i in range(0, df_activity_window.shape[1]-stride, stride):
        start = i
        end = start + stride
        print(start, end)
        df_a_w = df_activity_window.iloc[:, start:end]
        df_week = pd.concat([df_a_w, df_target], axis=1)
        print(df_week)
        process_data_frame_svm(
            svc_kernel,
            add_feature,
            meta_data,
            meta_data_short,
            output_dir / f"week_{cpt}",
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


if __name__ == "__main__":
    #typer.run(main)
    main(Path(f'E:/Data2/debug/test_distance_validation'), Path("E:/Data2/debug3/delmas/dataset4_mrnn_7day/activity_farmid_dbft_7_1min.csv"))
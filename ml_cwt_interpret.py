import glob
from pathlib import Path
from typing import List

import typer
from matplotlib.colors import LogNorm
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from cwt._cwt import CWT, DWT
from model.data_loader import load_activity_data, parse_param_from_filename
from preprocessing.preprocessing import apply_preprocessing_steps
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from utils.Utils import anscombe
import matplotlib.dates as mdates
import pycwt as wavelet
import pywt
from datetime import datetime, timedelta
from skfeature.function.similarity_based import fisher_score

from utils.visualisation import plot_high_dimension_db


def elemwise_cwt(X, preprocessing_steps, output_dir):
    f_transform = CWT(
        step_slug="_".join(preprocessing_steps),
        wavelet_f0=6,
        out_dir=output_dir,
        n_scales=8,
        sub_sample_scales=1,
        enable_coi=False,
        enable_graph_out=False,
    )
    X_cwt, _, _ = f_transform.transform(X)

    cwt_list_0 = []
    for cwt in X_cwt:
        cwt_list_0.append(
            np.reshape(cwt, (f_transform.shape[0], f_transform.shape[1]))
        )
    coefs_class0_mean = np.mean(cwt_list_0, axis=0)
    return coefs_class0_mean


def datetime_range(start, end, delta):
    current = start
    while current < end:
        yield current
        current += delta


def get_imp_stat(date_list, imp, activity, n_activity_days, X_train, output_dir):
    #print(date_list)
    light = np.array(["night" if x.hour>=7 and x.hour<19 else "day" for x in date_list])
    light_ = np.array([0 if x.hour>=7 and x.hour<19 else 1 for x in date_list])

    fig_, ax_ = plt.subplots(figsize=(12.80, 7.20))
    filename = f"{n_activity_days}_period_{X_train.shape[1]}.png"
    filepath = output_dir / filename
    print(filepath)
    ax_.plot(activity)
    ax_.plot(light_)
    fig_.savefig(filepath)

    day_imp = imp[light == "day"]
    night_imp = imp[light == "night"]

    std = [day_imp.std(), night_imp.std()]
    mean = [day_imp.mean(), night_imp.mean()]
    median = [np.median(day_imp), np.median(night_imp)]
    index = ['Daytime', 'Nighttime']
    df = pd.DataFrame({'std': std,
                       'mean': mean,
                       'median': median}, index=index)
    fig_ = df.plot.bar(rot=0, title="Feature importance", figsize=(3, 6)).get_figure()
    filename = f"{n_activity_days}_day_night_{X_train.shape[1]}.png"
    filepath = output_dir / filename
    print(filepath)
    fig_.savefig(filepath)


def main(
        output_dir: Path = typer.Option(
            ..., exists=False, file_okay=False, dir_okay=True, resolve_path=True
        ),
        dataset_folder: Path = typer.Option(
            ..., exists=True, file_okay=False, dir_okay=True, resolve_path=True
        ),
        class_healthy_label: List[str] = ["1To1"],
        class_unhealthy_label: List[str] = ["2To2"],
        preprocessing_steps: List[str] = ["None"],
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
        meta_col_str: List[str] = ["health", "label", "date"],
        roll_avg: int = 30,
        prct: int = 90,
        _size: int = 2,
        transform: str = "cwt",
        enable_graph_out: bool = True,
        # distance: bool = True,
        random_forest: bool = True,
        individual_to_ignore: List[str] = [],
        sampling: str = "T",
        resolution: float = None,
        plot_high_dimension_db: bool = False,
        p: bool = typer.Option(False, "--p"),
):
    """This script builds the graphs for cwt interpretation\n
    Args:\n
        output_dir: Output directory
        dataset_folder: Dataset input directory
        class_healthy: Label for healthy class
        class_unhealthy: Label for unhealthy class
        p: analyse famacha impact over time up to test date
    """

    files = glob.glob(str(dataset_folder / "*.csv"))  # find datset files
    print("found %d files." % len(files))
    print(files)

    for file in files:
        # days, farm_id, option, sampling = parse_param_from_filename(file)
        print(f"loading dataset file {file} ...")
        (
            data_frame,
            meta_data,
            meta_data_short,
            _,
            _,
            label_series,
            samples,
            _,
        ) = load_activity_data(
            output_dir,
            meta_columns,
            file,
            n_activity_days,
            class_healthy_label,
            class_unhealthy_label,
            imputed_days=n_imputed_days,
            preprocessing_steps=preprocessing_steps,
            meta_cols_str=meta_col_str,
            individual_to_ignore=individual_to_ignore,
            sampling=sampling,
            resolution=resolution
        )

        print(data_frame)

        data_frame_time, _, _ = apply_preprocessing_steps(
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
            farm_name="FARMS",
            keep_meta=False,
        )
        print(data_frame_time)

        data_frame_time_o, _, _ = apply_preprocessing_steps(
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
            ["QN", "ANSCOMBE", "LOG"],
            class_healthy_label,
            class_unhealthy_label,
            clf_name="SVM",
            n_scales=None,
            farm_name="FARMS",
            keep_meta=False,
        )


        # for n_activity_days in range(1, n_activity_days):
        print(n_activity_days)
        meta_ = meta_data_short[data_frame_time["health"].isin([0, 1])]
        data_frame_time = data_frame_time.loc[data_frame_time["health"].isin([0, 1])]

        data_frame = data_frame.loc[data_frame["health"].isin([0, 1])]

        X_train_o, y_train_o = (
            data_frame_time_o.iloc[:, : -2].values,
            data_frame_time_o["health"].values,
        )

        if n_activity_days > 0:
            X_train, y_train = (
                data_frame_time.iloc[:, : -2].values,
                data_frame_time["health"].values,
            )
        else:
            # for cats peak samples
            X_train, y_train = (
                data_frame_time.iloc[:, :-2].values,
                data_frame_time["health"].values,
            )
            n_activity_days = 1

        clf = SVC(kernel="linear", probability=True)
        #clf = LinearRegression()
        if random_forest:
            clf = RandomForestClassifier(random_state=0)

        print("fit...")
        fisher_s = fisher_score.fisher_score(X_train, y_train)
        clf.fit(X_train, y_train)

        if plot_high_dimension_db:
            plot_high_dimension_db(
                output_dir / "training",
                X_train,
                y_train,
                None,
                meta_,
                clf,
                n_activity_days,
                preprocessing_steps,
                0,
            )

        if random_forest:
            imp = clf.feature_importances_
        else:
            imp = abs(clf.coef_[0])

        y = clf.decision_function(X_train)
        w_norm = np.linalg.norm(clf.coef_)
        dist = abs(y / w_norm)

        df_dist = pd.DataFrame(X_train)
        df_dist["y"] = y_train
        df_dist["dist"] = dist
        df_dist = df_dist.sort_values("dist")

        df_dist_healthy = df_dist[df_dist["y"] == 0]
        df_dist_unhealthy = df_dist[df_dist["y"] == 1]

        d_m = max([df_dist_healthy.shape[0], df_dist_unhealthy.shape[0]])

        r = int(np.ceil(d_m / _size))

        fig, axs = plt.subplots(_size, 4, facecolor="white", figsize=(42.80, 12.80))
        cpt = 0
        min_a = []
        max_a = []
        for i in range(0, d_m, r):
            end = int(i + r)
            start = i
            d_h = df_dist_healthy[start:end]
            d_uh = df_dist_unhealthy[start:end]
            activity_h = np.mean(d_h.iloc[:, :-2]).values
            activity_uh = np.mean(d_uh.iloc[:, :-2]).values
            min_a.append(np.min(activity_h))
            min_a.append(np.min(activity_uh))
            max_a.append(np.max(activity_h))
            max_a.append(np.max(activity_uh))
        min_a = np.min(min_a)
        max_a = np.max(max_a)

        for i in range(0, d_m, r):
            end = int(i + r)
            start = i
            d = df_dist[start:end]

            min_d = d["dist"].values[0]
            max_d = d["dist"].values[-1]

            d_h = df_dist_healthy[start:end]
            d_uh = df_dist_unhealthy[start:end]

            activity_h = np.mean(d_h.iloc[:, :-2]).values
            activity_uh = np.mean(d_uh.iloc[:, :-2]).values

            cwt_h = elemwise_cwt(d_h.iloc[:, :-2].values, preprocessing_steps, output_dir)
            cwt_uh = elemwise_cwt(d_uh.iloc[:, :-2].values, preprocessing_steps, output_dir)

            if i == 0:
                mat_max = max([np.nanmax(cwt_h), np.nanmax(cwt_uh)])
                mat_min = min([np.nanmin(cwt_h), np.nanmin(cwt_uh)])


            axs[cpt, 0].plot(activity_h)
            axs[cpt, 0].set_title(f"Healthy samples distance range [{min_d:.3f} {max_d:.3f}] {d_h.iloc[:, :-2].shape}")
            axs[cpt, 0].set_xlabel("Time")
            axs[cpt, 0].set_ylabel("Activity")
            axs[cpt, 0].set_ylim([min_a, max_a])

            axs[cpt, 1].plot(activity_uh)
            axs[cpt, 1].set_title(f"Unhealthy samples distance range [{min_d:.3f} {max_d:.3f} {d_uh.iloc[:, :-2].shape}]")
            axs[cpt, 1].set_xlabel("Time")
            axs[cpt, 1].set_ylabel("Activity")
            axs[cpt, 1].set_ylim([min_a, max_a])

            origin = "upper"
            im = axs[cpt, 2].imshow(
                cwt_h,
                origin=origin,
                extent=[0, len(activity_h), 1, cwt_h.shape[0]],
                interpolation="nearest",
                aspect="auto",
                vmin=mat_min,
                vmax=mat_max
            )
            axs[cpt, 2].set_title(f"Element wise mean of cwt coefficients healthy")
            axs[cpt, 2].set_xlabel("Time")
            axs[cpt, 2].set_ylabel("Scales")
            fig.colorbar(im, ax=axs[cpt, 2])

            im = axs[cpt, 3].imshow(
                cwt_uh,
                origin=origin,
                extent=[0, len(activity_h), 1, cwt_h.shape[0]],
                interpolation="nearest",
                aspect="auto",
                vmin=mat_min,
                vmax=mat_max
            )
            axs[cpt, 3].set_title(f"Element wise mean of cwt coefficients unhealthy")
            axs[cpt, 3].set_xlabel("Time")
            axs[cpt, 3].set_ylabel("Scales")
            fig.colorbar(im, ax=axs[cpt, 3])
            cpt += 1

        filename = f"{n_activity_days}_{transform}_{r}_distance_from_db_{X_train.shape[1]}.png"
        filepath = output_dir / filename
        fig.tight_layout()
        print(filepath)
        fig.savefig(filepath)

        intercept = clf.intercept_
        mean_time = np.mean(X_train, axis=0)
        mean_time_o = np.mean(X_train_o, axis=0)

        date_list = [
            datetime(2016, 9, 1, 12) + timedelta(minutes=1 * x)
            for x in range(0, len(mean_time))
        ]

        get_imp_stat(date_list, imp, mean_time_o, n_activity_days, X_train, output_dir)

        fig, ax = plt.subplots(figsize=(16.80, 7.20))
        ax2 = ax.twinx()
        ax.plot(date_list, mean_time, label=f"mean activity of all samples({class_healthy_label}, {class_unhealthy_label}) after {preprocessing_steps}")
        ax.plot(date_list, mean_time_o, label=f"mean activity of all samples({class_healthy_label}, {class_unhealthy_label}) after {['QN', 'ANSCOMBE', 'LOG']}")
        # ax.plot(imp*mean, label="mean activity of all samples * feature importance")
        ax2.plot(date_list, imp, color="red", label="weight", alpha=0.3)
        #ax2.plot(date_list, fisher_s, color="green", label="Fisher score", alpha=0.3)

        #ax2.plot(date_list, intercept, color="purple", label="intercept", alpha=0.3)

        df_imp = pd.DataFrame(imp, columns=["imp"])
        rollavg = df_imp.imp.rolling(roll_avg).mean()
        ax2.plot(
            date_list,
            rollavg,
            color="black",
            label=f"feature importance rolling avg ({roll_avg} points)",
            alpha=0.9,
        )

        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")
        ax.set_title(f"Feature importance {type(clf).__name__} days={n_activity_days}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Activity")
        ax2.set_ylabel("importance", color="red")
        filename = f"{n_activity_days}_feature_importance_{X_train.shape[1]}.png"
        filepath = output_dir / filename
        print(filepath)

        T = 60*4
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%dT%H %p"))
        ax2.xaxis.set_major_locator(mdates.MinuteLocator(interval=T*n_activity_days))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%dT%H %p"))
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=T*n_activity_days))
        fig.autofmt_xdate()
        fig.savefig(filepath)

        if transform == "dwt":
            f_transform = DWT(
                dwt_window="coif1",
                step_slug="_".join(preprocessing_steps),
                out_dir=output_dir,
                enable_graph_out=enable_graph_out,
            )
            _, X_train = f_transform.transform(X_train)
        if transform == "cwt":
            f_transform = CWT(
                step_slug="_".join(preprocessing_steps),
                wavelet_f0=6,
                out_dir=output_dir,
                n_scales=8,
                sub_sample_scales=1,
                enable_coi=False,
                enable_graph_out=enable_graph_out,
            )
            X_train, _, _ = f_transform.transform(X_train)
        X_train_o = X_train.copy()
        X_train[np.isnan(X_train)] = -1
        # scales = CWT_Transform.get_scales()
        clf = SVC(kernel="linear", probability=True)
        #clf = LinearRegression()
        if random_forest:
            clf = RandomForestClassifier(random_state=0)

        print("fit...")
        clf.fit(X_train, y_train)
        if random_forest:
            imp = clf.feature_importances_
        else:
            imp = abs(clf.coef_[0])

        # if distance:
        #     y = clf.decision_function(X_train)
        #     w_norm = np.linalg.norm(clf.coef_)
        #     dist = y / w_norm
        #     imp = dist
        #
        intercept = clf.intercept_
        imp_top_n_perct = imp.copy()
        imp_top_n_perct[
            imp_top_n_perct <= np.percentile(imp_top_n_perct, prct)
            ] = np.nan

        mean_ = np.mean(X_train, axis=0)

        cwt_0 = X_train[y_train == 0]
        cwt_1 = X_train[y_train == 1]

        cwt_list_0 = []
        for cwt in cwt_0:
            cwt_list_0.append(
                np.reshape(cwt, (f_transform.shape[0], f_transform.shape[1]))
            )

        cwt_list_1 = []
        for cwt in cwt_1:
            cwt_list_1.append(
                np.reshape(cwt, (f_transform.shape[0], f_transform.shape[1]))
            )

        coefs_class0_mean = np.mean(cwt_list_0, axis=0)
        coefs_class1_mean = np.mean(cwt_list_1, axis=0)
        cwt_imp = np.reshape(imp, (f_transform.shape[0], f_transform.shape[1]))
        cwt_imp_top = np.reshape(
            imp_top_n_perct, (f_transform.shape[0], f_transform.shape[1])
        )

        fig, axs = plt.subplots(3, 1, facecolor="white", figsize=(12.80, 18.80))
        axs = axs.ravel()
        fig, ax = plt.subplots(figsize=(12.80, 7.20))
        ax2 = ax.twinx()
        ax.plot(mean_, label=f"mean {transform}(flatten) of all samples")
        # ax.plot(imp*mean, label="mean activity of all samples * feature importance")
        ax2.plot(imp, color="red", label="feature importance", alpha=0.3)
        df_imp = pd.DataFrame(imp, columns=["imp"])
        rollavg = df_imp.imp.rolling(1000).mean()
        ax2.plot(
            rollavg,
            color="black",
            label=f"feature importance rolling avg ({1000} points)",
            alpha=0.9,
        )

        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")
        ax.set_title(f"Feature importance {type(clf).__name__} days={n_activity_days}")
        ax.set_xlabel(f"{transform} (features)")
        ax.set_ylabel("Activity")
        ax2.set_ylabel("importance", color="red")
        filename = (
            f"{n_activity_days}_{transform}_feature_importance_{X_train.shape[1]}.png"
        )
        filepath = output_dir / filename
        print(filepath)
        fig.savefig(filepath)

        cwt_0 = X_train[y_train == 0]
        cwt_1 = X_train[y_train == 1]

        cwt_list_0 = []
        for cwt in cwt_0:
            # iwave_test = wavelet.icwt(np.reshape(cwt, (f_transform.shape[0], f_transform.shape[1])), f_transform.scales, f_transform.delta_t,
            #                        wavelet=f_transform.wavelet_type.lower()).real
            # plt.plot(iwave_test)
            # plt.show()

            cwt_list_0.append(
                np.reshape(cwt, (f_transform.shape[0], f_transform.shape[1]))
            )

        cwt_list_1 = []
        for cwt in cwt_1:
            cwt_list_1.append(
                np.reshape(cwt, (f_transform.shape[0], f_transform.shape[1]))
            )

        coi_mask = np.reshape(
            X_train_o[0], (f_transform.shape[0], f_transform.shape[1])
        )
        coefs_class0_mean = np.mean(cwt_list_0, axis=0)
        coefs_class0_mean[np.isnan(coi_mask)] = np.nan
        coefs_class1_mean = np.mean(cwt_list_1, axis=0)
        coefs_class1_mean[np.isnan(coi_mask)] = np.nan
        cwt_imp = np.reshape(imp, (f_transform.shape[0], f_transform.shape[1]))
        # cwt_intercept = np.reshape(intercept, (f_transform.shape[0], f_transform.shape[1]))

        #cwt_imp[np.isnan(coi_mask)] = np.nan
        cwt_imp_top = np.reshape(
            imp_top_n_perct, (f_transform.shape[0], f_transform.shape[1])
        )
        cwt_imp_top[np.isnan(coi_mask)] = np.nan

        fig, axs = plt.subplots(5, 2, facecolor="white", figsize=(28.60, 26.80))
        origin = "upper"
        if transform == "dwt":
            fig, axs = plt.subplots(3, 2, facecolor="white", figsize=(28.60, 12.80))
            origin = "lower"
        axs = axs.ravel()

        # axs[0].pcolormesh(
        #     np.arange(coefs_class0_mean.shape[1]),
        #     scales,
        #     coefs_class0_mean,
        #     cmap="viridis"
        # )
        mat_max = max([np.nanmax(coefs_class0_mean), np.nanmax(coefs_class1_mean)])
        mat_min = min([np.nanmin(coefs_class0_mean), np.nanmin(coefs_class1_mean)])
        date_list = mdates.date2num(date_list)
        im = axs[0].imshow(
            coefs_class0_mean,
            origin=origin,
            extent=[date_list[0], date_list[-1], 1, coefs_class0_mean.shape[0]],
            interpolation="nearest",
            aspect="auto",
            vmin=mat_min,
            vmax=mat_max
        )
        fig.colorbar(im, ax=axs[0])
        date_format = "%dT%H %p"
        if n_activity_days < 0:
            date_format = "00:%H"
        axs[0].xaxis.set_major_formatter(mdates.DateFormatter(date_format))
        axs[0].xaxis.set_major_locator(mdates.MinuteLocator(interval=T*n_activity_days))
        axs[0].set_title(f"Element wise mean of {transform} coefficients healthy")
        axs[0].set_xlabel("Time")
        axs[0].set_ylabel("Scales")

        im = axs[1].imshow(
            coefs_class1_mean,
            origin=origin,
            extent=[date_list[0], date_list[-1], 1, coefs_class0_mean.shape[0]],
            interpolation="nearest",
            aspect="auto",
            vmin=mat_min,
            vmax=mat_max
        )
        fig.colorbar(im, ax=axs[1])
        axs[1].xaxis.set_major_formatter(mdates.DateFormatter(date_format))
        axs[1].xaxis.set_major_locator(mdates.MinuteLocator(interval=T*n_activity_days))
        axs[1].set_title(f"Element wise mean of {transform} coefficients unhealthy")
        axs[1].set_xlabel("Time")
        axs[1].set_ylabel("Scales")

        mat_max = max([np.nanmax(cwt_imp), np.nanmax(cwt_imp_top)])
        mat_min = min([np.nanmin(cwt_imp), np.nanmin(cwt_imp_top)])
        im = axs[2].imshow(
            cwt_imp,
            origin=origin,
            extent=[date_list[0], date_list[-1], 1, coefs_class0_mean.shape[0]],
            interpolation="nearest",
            aspect="auto",
            vmin=mat_min,
            vmax=mat_max
        )
        fig.colorbar(im, ax=axs[2])
        axs[2].xaxis.set_major_formatter(mdates.DateFormatter(date_format))
        axs[2].xaxis.set_major_locator(mdates.MinuteLocator(interval=T*n_activity_days))
        axs[2].set_title(f"{transform} Features importance {type(clf).__name__}")
        axs[2].set_xlabel("Time")
        axs[2].set_ylabel("Scales")

        im = axs[3].imshow(
            cwt_imp_top,
            origin=origin,
            extent=[date_list[0], date_list[-1], 1, coefs_class0_mean.shape[0]],
            interpolation="nearest",
            aspect="auto",
            vmin=mat_min,
            vmax=mat_max
        )
        fig.colorbar(im, ax=axs[3])
        axs[3].xaxis.set_major_formatter(mdates.DateFormatter(date_format))
        axs[3].xaxis.set_major_locator(mdates.MinuteLocator(interval=T*n_activity_days))
        axs[3].set_title(
            f"{transform} Features importance top 10% {type(clf).__name__} days={n_activity_days}"
        )
        axs[3].set_xlabel("Time")
        axs[3].set_ylabel("Scales")

        a = (cwt_imp * coefs_class0_mean) - intercept
        b = (cwt_imp * coefs_class1_mean) - intercept
        # if distance:
        #     a = (cwt_imp * coefs_class0_mean)
        #     b = (cwt_imp * coefs_class1_mean)

        mat_max = max([np.nanmax(a), np.nanmax(b)])
        mat_min = min([np.nanmin(a), np.nanmin(b)])
        im = axs[4].imshow(
            a,
            origin=origin,
            extent=[date_list[0], date_list[-1], 1, coefs_class0_mean.shape[0]],
            interpolation="nearest",
            aspect="auto",
            vmin=mat_min,
            vmax=mat_max
        )
        fig.colorbar(im, ax=axs[4])
        axs[4].xaxis.set_major_formatter(mdates.DateFormatter(date_format))
        axs[4].xaxis.set_major_locator(mdates.MinuteLocator(interval=T*n_activity_days))
        axs[4].set_title(
            f"{transform} Features importance multipied by coef of healthy class days={n_activity_days}"
        )
        axs[4].set_xlabel("Time")
        axs[4].set_ylabel("Scales")

        im = axs[5].imshow(
            b,
            origin=origin,
            extent=[date_list[0], date_list[-1], 1, coefs_class0_mean.shape[0]],
            interpolation="nearest",
            aspect="auto",
            vmin=mat_min,
            vmax=mat_max
        )
        fig.colorbar(im, ax=axs[5])
        axs[5].xaxis.set_major_formatter(mdates.DateFormatter(date_format))
        axs[5].xaxis.set_major_locator(mdates.MinuteLocator(interval=T*n_activity_days))
        axs[5].set_title(
            f"{transform} Features importance multipied by coef of unhealthy class days={n_activity_days}"
        )
        axs[5].set_xlabel("Time")
        axs[5].set_ylabel("Scales")
        #########################################

        if transform == "cwt":
            iwave_h = wavelet.icwt(coefs_class0_mean, f_transform.scales, f_transform.delta_t,
                                   wavelet=f_transform.wavelet_type.lower()).real
            iwave_uh = wavelet.icwt(coefs_class1_mean, f_transform.scales, f_transform.delta_t,
                                    wavelet=f_transform.wavelet_type.lower()).real
            ymin = min([iwave_h.min(), iwave_uh.min()])
            ymax = max([iwave_h.max(), iwave_uh.max()])

            axs[6].plot(iwave_h)
            fig.colorbar(im, ax=axs[6])
            #axs[6].xaxis.set_major_formatter(mdates.DateFormatter(date_format))
            # axs[6].xaxis.set_major_locator(mdates.MinuteLocator(interval=T * n_activity_days))
            axs[6].set_title(
                f"{transform} Inverse of coefs of healthy d={n_activity_days}"
            )
            axs[6].set_xlabel("Time")
            axs[6].set_ylabel("Activity")
            axs[6].set_ylim([ymin, ymax])

            axs[7].plot(iwave_uh)
            fig.colorbar(im, ax=axs[7])
            #axs[7].xaxis.set_major_formatter(mdates.DateFormatter(date_format))
            # axs[7].xaxis.set_major_locator(mdates.MinuteLocator(interval=T * n_activity_days))
            axs[7].set_title(
                f"{transform} Inverse of coefs of healthy d={n_activity_days}"
            )
            axs[7].set_xlabel("Time")
            axs[7].set_ylabel("Activity")
            axs[7].set_ylim([ymin, ymax])

            iwave_h = abs(wavelet.icwt(a, f_transform.scales, f_transform.delta_t,
                                   wavelet=f_transform.wavelet_type.lower()))
            iwave_uh = abs(wavelet.icwt(b, f_transform.scales, f_transform.delta_t,
                                    wavelet=f_transform.wavelet_type.lower()))
            ymin = min([iwave_h.min(), iwave_uh.min()])
            ymax = max([iwave_h.max(), iwave_uh.max()])

            axs[8].plot(iwave_h)
            fig.colorbar(im, ax=axs[8])
            #axs[8].xaxis.set_major_formatter(mdates.DateFormatter(date_format))
            #axs[8].xaxis.set_major_locator(mdates.MinuteLocator(interval=T * n_activity_days))
            axs[8].set_title(
                f"{transform} Inverse of Features importance multipied by coef of healthy d={n_activity_days}"
            )
            axs[8].set_xlabel("Time")
            axs[8].set_ylabel("Activity")
            axs[8].set_ylim([ymin, ymax])

            axs[9].plot(iwave_uh)
            fig.colorbar(im, ax=axs[9])
            #axs[9].xaxis.set_major_formatter(mdates.DateFormatter(date_format))
            #axs[9].xaxis.set_major_locator(mdates.MinuteLocator(interval=T * n_activity_days))
            axs[9].set_title(
                f"{transform} Inverse of Features importance multipied by coef of healthy d={n_activity_days}"
            )
            axs[9].set_xlabel("Time")
            axs[9].set_ylabel("Activity")
            axs[9].set_ylim([ymin, ymax])

        # if transform == "dwt":
        #     iwave_h = pywt.waverec(cwt_imp*coefs_class0_mean, f_transform.wavelet, f_transform.mode)
        #     iwave_uh = pywt.waverec(cwt_imp*coefs_class1_mean, f_transform.wavelet, f_transform.mode)
        #
        #     ymin = min([iwave_h.min(), iwave_uh.min()])
        #     ymax = max([iwave_h.max(), iwave_uh.max()])
        #
        #     axs[6].plot(iwave_h)
        #     fig.colorbar(im, ax=axs[6])
        #     # axs[6].xaxis.set_major_formatter(mdates.DateFormatter(date_format))
        #     # axs[6].xaxis.set_major_locator(mdates.MinuteLocator(interval=T * n_activity_days))
        #     axs[6].set_title(
        #         f"{transform} Inverse of coefs of healthy d={n_activity_days}"
        #     )
        #     axs[6].set_xlabel("Time")
        #     axs[6].set_ylabel("Activity")
        #     axs[6].set_ylim([ymin, ymax])
        #
        #     axs[7].plot(iwave_uh)
        #     fig.colorbar(im, ax=axs[7])
        #     # axs[7].xaxis.set_major_formatter(mdates.DateFormatter(date_format))
        #     # axs[7].xaxis.set_major_locator(mdates.MinuteLocator(interval=T * n_activity_days))
        #     axs[7].set_title(
        #         f"{transform} Inverse of coefs of healthy d={n_activity_days}"
        #     )
        #     axs[7].set_xlabel("Time")
        #     axs[7].set_ylabel("Activity")
        #     axs[7].set_ylim([ymin, ymax])
        #
        #     iwave_h = pywt.waverec(cwt_imp*coefs_class0_mean, f_transform.wavelet, f_transform.mode)
        #     iwave_uh = pywt.waverec(cwt_imp*coefs_class1_mean, f_transform.wavelet, f_transform.mode)
        #     ymin = min([iwave_h.min(), iwave_uh.min()])
        #     ymax = max([iwave_h.max(), iwave_uh.max()])
        #     im = axs[8].plot(iwave_h)
        #     fig.colorbar(im, ax=axs[8])
        #     # axs[6].xaxis.set_major_formatter(mdates.DateFormatter(date_format))
        #     # axs[6].xaxis.set_major_locator(mdates.MinuteLocator(interval=T * n_activity_days))
        #     axs[8].set_title(
        #         f"{transform} Inverse of Features importance multipied by coef of healthy d={n_activity_days}"
        #     )
        #     axs[8].set_xlabel("Time")
        #     axs[8].set_ylabel("Activity")
        #     axs[8].set_ylim([ymin, ymax])
        #
        #     axs[9].plot(iwave_uh)
        #     fig.colorbar(im, ax=axs[9])
        #     # axs[7].xaxis.set_major_formatter(mdates.DateFormatter(date_format))
        #     # axs[7].xaxis.set_major_locator(mdates.MinuteLocator(interval=T * n_activity_days))
        #     axs[9].set_title(
        #         f"{transform} Inverse of Features importance multipied by coef of healthy d={n_activity_days}"
        #     )
        #     axs[9].set_xlabel("Time")
        #     axs[9].set_ylabel("Activity")
        #     axs[9].set_ylim([ymin, ymax])

        filename = f"{n_activity_days}_{transform}_reshaped_feature_importance_{X_train.shape[1]}.png"
        filepath = output_dir / filename
        # fig.autofmt_xdate()
        fig.tight_layout()
        print(filepath)
        fig.savefig(filepath)
        # axs[0].pcolormesh(
        #     np.arange(coefs_class0_mean.shape[1]),
        #     scales,
        #     coefs_class0_mean,
        #     cmap="viridis"
        # )
        # axs[0].imshow(
        #     coefs_class0_mean, extent=[0, coefs_class0_mean.shape[1], coefs_class0_mean.shape[0], 1],
        #     interpolation="nearest", aspect='auto'
        # )
        # axs[0].set_title("Element wise mean of CWT healthy")
        # axs[0].set_xlabel('Time')
        # axs[0].set_ylabel('Scales')
        #
        # axs[1].imshow(
        #     coefs_class1_mean, extent=[0, coefs_class0_mean.shape[1], coefs_class0_mean.shape[0], 1],
        #     interpolation="nearest", aspect='auto'
        # )
        # axs[1].set_title("Element wise mean of CWT unhealthy")
        # axs[1].set_xlabel('Time')
        # axs[1].set_ylabel('Scales')
        #
        # axs[2].imshow(
        #     cwt_imp, extent=[0, coefs_class0_mean.shape[1], coefs_class0_mean.shape[0], 1],
        #     interpolation="nearest", aspect='auto'
        # )
        # axs[2].set_title(f"CWT Features importance {type(clf).__name__}")
        # axs[2].set_xlabel('Time')
        # axs[2].set_ylabel('Scales')
        #
        # filename = f"cwt_reshaped_feature_importance_{X_train.shape[1]}.png"
        # filepath = output_dir / filename
        # fig.tight_layout()
        # print(filepath)
        # fig.savefig(filepath)

        # df_cwt, class0_count, class1_count, cwt_coefs_data, features_names = get_cwt_data_frame(data_frame)
        # if p:
        #     dfs, data = chunck_df(n_activity_days, df_cwt, cwt_coefs_data)
        # else:
        #     dfs, data = process_df(df_cwt, cwt_coefs_data)
        #
        # explain_cwt(
        #     n_activity_days,
        #     dfs,
        #     data,
        #     output_dir,
        #     class0_count,
        #     class1_count,
        #     label_series,
        #     features_names
        # )
        #
        # if p:
        #     plot_progression(output_dir, n_activity_days)


if __name__ == "__main__":
    # typer.run(main)

    for t in ["dwt"]:
        for j in [1, 2, 3, 4, 5, 6, 7]:
            main(
                Path(f"E:/preprint/thesis/interpret/delmas_{t}_explain_{j}_datasetmrnn7_17"),
                Path("E:/thesis/datasets/delmas/datasetmrnn7_17"),
                preprocessing_steps=["QN", "ANSCOMBE", "LOG", "STDS"],
                p=False,
                n_activity_days=j,
                transform=t,
                enable_graph_out=False,
                random_forest=False
            )
            main(
                Path(f"E:/preprint/thesis/interpret/cedara_{t}_explain_{j}_datasetmrnn7_23"),
                Path("E:/thesis/datasets/delmas/datasetmrnn7_23"),
                preprocessing_steps=["QN", "ANSCOMBE", "LOG", "STDS"],
                p=False,
                n_activity_days=j,
                transform=t,
                enable_graph_out=False,
                random_forest=False
            )

            # main(
            #     Path(f"E:/Data2/debug_/{t}_explain_{j}/without_resampling"),
            #     Path("E:/Data2/debug3/delmas/dataset4_mrnn_7day"),
            #     p=False,
            #     n_activity_days=j,
            #     transform=t,
            #     resolution=None,
            #     enable_graph_out=True
            # )

        # main(
        #     Path(f"E:/Data2/debug/{t}_cat_explain_2"),
        #     Path("E:/Cats/build_multiple_peak_permutations_4/004__0_00100__120/dataset/training_sets/samples"),
        #     p=False,
        #     n_activity_days=-1,
        #     n_imputed_days=-1,
        #     transform=t,
        #     meta_columns=[
        #         "label",
        #         "id",
        #         "imputed_days",
        #         "date",
        #         "health",
        #         "target",
        #         "age",
        #         "name",
        #         "mobility_score",
        #     ],
        #     preprocessing_steps=["QN", "STD"],
        #     meta_col_str=[],
        #     individual_to_ignore=["MrDudley", "Oliver_F", "Lucy"],
        #     class_healthy_label=["0.0"],
        #     class_unhealthy_label=["1.0"],
        # )
        #
        # main(
        #     Path(f"E:/Data2/debug/{t}_cat_explain_3"),
        #     Path("E:/Cats/build_multiple_peak_permutations_4/004__0_00100__120/dataset/training_sets/samples"),
        #     p=False,
        #     n_activity_days=-1,
        #     n_imputed_days=-1,
        #     transform=t,
        #     meta_columns=[
        #         "label",
        #         "id",
        #         "imputed_days",
        #         "date",
        #         "health",
        #         "target",
        #         "age",
        #         "name",
        #         "mobility_score",
        #     ],
        #     preprocessing_steps=["QN", "ANSCOMBE", "LOG"],
        #     meta_col_str=[],
        #     individual_to_ignore=["MrDudley", "Oliver_F", "Lucy"],
        #     class_healthy_label=["0.0"],
        #     class_unhealthy_label=["1.0"],
        # )

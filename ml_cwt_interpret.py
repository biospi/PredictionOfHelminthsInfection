import glob
from pathlib import Path
from typing import List

import typer
from matplotlib.colors import LogNorm
from sklearn.svm import SVC

from cwt._cwt import CWT
from model.data_loader import load_activity_data, parse_param_from_filename
from preprocessing.preprocessing import apply_preprocessing_steps
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main(
    output_dir: Path = typer.Option(
        ..., exists=False, file_okay=False, dir_okay=True, resolve_path=True
    ),
    dataset_folder: Path = typer.Option(
        ..., exists=True, file_okay=False, dir_okay=True, resolve_path=True
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
    meta_col_str: List[str] = ["health", "label", "date"],
    roll_avg: int = 30,
    p: bool = typer.Option(False, "--p")
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
        #days, farm_id, option, sampling = parse_param_from_filename(file)
        print(f"loading dataset file {file} ...")
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
            file,
            n_activity_days,
            class_healthy_label,
            class_unhealthy_label,
            imputed_days=n_imputed_days,
            preprocessing_steps=preprocessing_steps,
            meta_cols_str=meta_col_str
        )

        print(data_frame)

        data_frame_time, _ = apply_preprocessing_steps(
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

        for n_activity_days in range(1, n_activity_days):
            data_frame_time = data_frame_time.loc[data_frame_time['health'].isin([0, 1])]
            X_train, y_train = data_frame_time.iloc[:, -1440*n_activity_days:-2].values, data_frame_time["health"].values
            clf = SVC(kernel="linear", probability=True)
            print("fit...")
            clf.fit(X_train, y_train)
            imp = np.abs(clf.coef_[0])
            mean_time = np.mean(X_train, axis=0)

            fig, ax = plt.subplots()
            ax2 = ax.twinx()
            ax.plot(mean_time, label="mean activity of all samples")
            #ax.plot(imp*mean, label="mean activity of all samples * feature importance")
            ax2.plot(imp, color="red", label="feature importance", alpha=0.3)

            df_imp = pd.DataFrame(imp, columns=["imp"])
            rollavg = df_imp.imp.rolling(roll_avg).mean()
            ax2.plot(rollavg, color="black", label=f"feature importance rolling avg ({roll_avg} points)", alpha=0.9)

            ax.legend(loc="upper left")
            ax2.legend(loc="upper right")
            ax.set_title(f"Feature importance {type(clf).__name__}")
            ax.set_xlabel('Time (features)')
            ax.set_ylabel('Activity')
            ax2.set_ylabel('Importance', color='red')
            filename = f"feature_importance_{X_train.shape[1]}.png"
            filepath = output_dir / filename
            print(filepath)
            fig.savefig(filepath)

            CWT_Transform = CWT(
                step_slug='_'.join(preprocessing_steps),
                wavelet_f0=6,
                out_dir=output_dir,
                n_scales=8,
                sub_sample_scales=1,
                enable_coi=True,
                enable_graph_out=False
            )
            X_train, _, _ = CWT_Transform.transform(X_train)
            X_train[np.isnan(X_train)] = -1
            scales = CWT_Transform.get_scales()
            clf = SVC(kernel="linear", probability=True)
            print("fit...")
            clf.fit(X_train, y_train)
            imp = np.abs(clf.coef_[0])
            mean_cwt = np.mean(X_train, axis=0)

            fig, ax = plt.subplots()
            ax2 = ax.twinx()
            ax.plot(mean_cwt, label="mean cwt(flatten) of all samples")
            #ax.plot(imp*mean, label="mean activity of all samples * feature importance")
            ax2.plot(imp, color="red", label="feature importance", alpha=0.3)
            df_imp = pd.DataFrame(imp, columns=["imp"])
            rollavg = df_imp.imp.rolling(1000).mean()
            ax2.plot(rollavg, color="black", label=f"feature importance rolling avg ({1000} points)", alpha=0.9)

            ax.legend(loc="upper left")
            ax2.legend(loc="upper right")
            ax.set_title(f"Feature importance {type(clf).__name__}")
            ax.set_xlabel('CWT (features)')
            ax.set_ylabel('Activity')
            ax2.set_ylabel('Importance', color='red')
            filename = f"cwt_feature_importance_{X_train.shape[1]}.png"
            filepath = output_dir / filename
            print(filepath)
            fig.savefig(filepath)

            cwt_0 = X_train[y_train == 0]
            cwt_1 = X_train[y_train == 1]

            cwt_list_0 = []
            for cwt in cwt_0:
                cwt_list_0.append(np.reshape(cwt, (len(scales), len(mean_time))))

            cwt_list_1 = []
            for cwt in cwt_1:
                cwt_list_1.append(np.reshape(cwt, (len(scales), len(mean_time))))

            coefs_class0_mean = np.mean(cwt_list_0, axis=0)
            coefs_class1_mean = np.mean(cwt_list_1, axis=0)
            cwt_imp = np.reshape(imp, (len(scales), len(mean_time)))

            fig, axs = plt.subplots(3, 1, facecolor='white')
            axs = axs.ravel()

            # axs[0].pcolormesh(
            #     np.arange(coefs_class0_mean.shape[1]),
            #     scales,
            #     coefs_class0_mean,
            #     cmap="viridis"
            # )
            axs[0].imshow(
                coefs_class0_mean, extent=[0, coefs_class0_mean.shape[1], coefs_class0_mean.shape[0], 1],
                interpolation="nearest", norm=LogNorm(vmin=0.01, vmax=1)
            )
            axs[0].set_title("Element wise mean of CWT healthy")
            axs[0].set_xlabel('Time')
            axs[0].set_ylabel('Scales')

            axs[1].imshow(
                coefs_class1_mean, extent=[0, coefs_class0_mean.shape[1], coefs_class0_mean.shape[0], 1],
                interpolation="nearest", norm=LogNorm(vmin=0.01, vmax=1)
            )
            axs[1].set_title("Element wise mean of CWT unhealthy")
            axs[1].set_xlabel('Time')
            axs[1].set_ylabel('Scales')

            axs[2].imshow(
                cwt_imp, extent=[0, coefs_class0_mean.shape[1], coefs_class0_mean.shape[0], 1],
                interpolation="nearest"
            )
            axs[2].set_title(f"CWT Features importance {type(clf).__name__}")
            axs[2].set_xlabel('Time')
            axs[2].set_ylabel('Scales')

            filename = f"cwt_reshaped_feature_importance_{X_train.shape[1]}.png"
            filepath = output_dir / filename
            fig.tight_layout()
            print(filepath)
            fig.savefig(filepath)


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
    #typer.run(main)
    main(Path(f'E:/Data2/debug/cwt_explain'), Path("E:/Data2/debug3/delmas/dataset4_mrnn_7day"), p=False)
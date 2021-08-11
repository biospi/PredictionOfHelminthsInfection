import warnings
import pandas as pd
import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from cwt._cwt import STFT, CWT, CWTVisualisation
from data_imputation.model_utils import MinMaxScaler

from utils._anscombe import Anscombe, Sqrt, Log
from utils._normalisation import BaseLineScaler, CenterScaler, QuotientNormalizer
from utils.visualisation import plotDistribution


def setupGraphOutputPath(output_dir):
    graph_outputdir = output_dir / "input_graphs"
    # if os.path.exists(graph_outputdir):
    #     print("purge %s ..." % graph_outputdir)
    #     try:
    #         shutil.rmtree(graph_outputdir)
    #     except IOError:
    #         print("file not found.")
    graph_outputdir.mkdir(parents=True, exist_ok=True)
    return graph_outputdir


def apply_preprocessing_steps(
    days,
    df_hum,
    df_temp,
    sfft_window,
    wavelet_f0,
    animal_ids,
    df,
    N_META,
    output_dir,
    steps,
    class_healthy_label,
    class_unhealthy_label,
    class_healthy_target,
    class_unhealthy_target,
    clf_name="",
    output_dim=2,
    n_scales=None,
    farm_name="",
    keep_meta=False
):
    step_slug = "_".join(steps)
    step_slug = farm_name + "_" + step_slug
    graph_outputdir = setupGraphOutputPath(output_dir) / clf_name / step_slug

    if len(steps) == 0:
        print("no steps to apply! return data as is")
        return df
    print("BEFORE STEP ->", df)
    # plotDistribution(df.iloc[:, :-N_META].values, graph_outputdir, "data_distribution_before_%s" % step_slug)
    for step in steps:
        if step not in [
            "ANSCOMBE",
            "LOG",
            "QN",
            "CWT",
            "CENTER",
            "MINMAX",
            "PCA",
            "BASELINERM",
            "STFT",
            "STANDARDSCALER",
            "DIFFAPPEND",
            "DIFFLASTD",
            "DIFF",
            "DIFFLASTDAPPEND",
            "TSNE",
        ]:
            warnings.warn("processing step %s does not exist!" % step)
        # plotDistribution(df.iloc[:, :-N_META].values, graph_outputdir, "data_distribution_before_%s" % step)
        print("applying STEP->%s in [%s]..." % (step, step_slug.replace("_", "->")))
        if step == "TEMPERATUREAPPEND":
            df_activity = df.copy().iloc[:, :-N_META]
            df_activity.index = df.index  # need to keep original sample index!!!!
            df_meta = df.iloc[:, -N_META:]
            df_temp = df_temp.loc[df.index]
            df = pd.concat([df_activity, df_temp, df_meta], axis=1)
            new_header = [str(x) for x in np.arange(df.shape[1] - N_META)] + df.columns[
                df.shape[1] - N_META :
            ].tolist()
            df.columns = new_header

        if step == "HUMIDITYAPPEND":
            df_activity = df.copy().iloc[:, :-N_META]
            df_activity.index = df.index  # need to keep original sample index!!!!
            df_meta = df.iloc[:, -N_META:]
            df_hum = df_hum.loc[df.index]
            df = pd.concat([df_activity, df_hum, df_meta], axis=1)
            new_header = [str(x) for x in np.arange(df.shape[1] - N_META)] + df.columns[
                df.shape[1] - N_META :
            ].tolist()
            df.columns = new_header

        if step == "DIFFAPPEND":
            df_activity = df.copy().iloc[:, :-N_META]
            df_meta = df.iloc[:, -N_META:]
            df_diff = pd.DataFrame(
                df_activity.copy().iloc[:, 1440:].values
                - df_activity.copy().iloc[:, :-1440].values
            )
            df = pd.concat(
                [
                    df_activity.reset_index(drop=True),
                    df_diff.reset_index(drop=True),
                    df_meta.reset_index(drop=True),
                ],
                axis=1,
            )
            new_header = [str(x) for x in np.arange(df.shape[1] - N_META)] + df.columns[
                df.shape[1] - N_META :
            ].tolist()
            df.columns = new_header
            df.index = df_activity.index

        if step == "DIFF":
            df_activity = df.copy().iloc[:, :-N_META]
            df_meta = df.iloc[:, -N_META:]
            df_diff = pd.DataFrame(
                df_activity.copy().iloc[:, 1440:].values
                - df_activity.copy().iloc[:, :-1440].values
            )
            df = pd.concat(
                [df_diff.reset_index(drop=True), df_meta.reset_index(drop=True)], axis=1
            )
            new_header = [str(x) for x in np.arange(df.shape[1] - N_META)] + df.columns[
                df.shape[1] - N_META :
            ].tolist()
            df.columns = new_header
            df.index = df_activity.index

        if step == "DIFFLASTD":
            df_activity = df.copy().iloc[:, :-N_META]
            df_meta = df.iloc[:, -N_META:]

            df_last_day = df_activity.copy().iloc[:, -1440:]
            df_last_day = pd.concat([df_last_day] * (days - 1), axis=1)

            df_to_sub = df_activity.copy().iloc[:, :-1441]

            df_diff = pd.DataFrame(df_to_sub.values - df_last_day.values)
            df = pd.concat(
                [df_diff.reset_index(drop=True), df_meta.reset_index(drop=True)], axis=1
            )
            new_header = [str(x) for x in np.arange(df.shape[1] - N_META)] + df.columns[
                df.shape[1] - N_META :
            ].tolist()
            df.columns = new_header
            df.index = df_activity.index

        if step == "DIFFLASTDAPPEND":
            df_activity = df.copy().iloc[:, :-N_META]
            df_meta = df.iloc[:, -N_META:]

            df_last_day = df_activity.copy().iloc[:, -1440:]
            df_last_day = pd.concat([df_last_day] * (days - 1), axis=1)

            df_to_sub = df_activity.copy().iloc[:, :-1441]

            df_diff = pd.DataFrame(df_to_sub.values - df_last_day.values)
            df = pd.concat(
                [
                    df_activity.reset_index(drop=True),
                    df_diff.reset_index(drop=True),
                    df_meta.reset_index(drop=True),
                ],
                axis=1,
            )
            new_header = [str(x) for x in np.arange(df.shape[1] - N_META)] + df.columns[
                df.shape[1] - N_META :
            ].tolist()
            df.columns = new_header
            df.index = df_activity.index

        if step == "TEMPERATURE":
            df_activity = df.copy().iloc[:, :-N_META]
            df_activity.index = df.index  # need to keep original sample index!!!!
            df_meta = df.iloc[:, -N_META:]
            df_temp = df_temp.loc[df.index]
            df = pd.concat([df_temp, df_meta], axis=1)
            new_header = [str(x) for x in np.arange(df.shape[1] - N_META)] + df.columns[
                df.shape[1] - N_META :
            ].tolist()
            df.columns = new_header

        if step == "HUMIDITY":
            df_activity = df.copy().iloc[:, :-N_META]
            df_activity.index = df.index  # need to keep original sample index!!!!
            df_meta = df.iloc[:, -N_META:]
            df_hum = df_hum.loc[df.index]
            df = pd.concat([df_hum, df_meta], axis=1)
            new_header = [str(x) for x in np.arange(df.shape[1] - N_META)] + df.columns[
                df.shape[1] - N_META :
            ].tolist()
            df.columns = new_header

        if step == "BASELINERM":
            df.iloc[:, :-N_META] = BaseLineScaler().fit_transform(
                df.iloc[:, :-N_META].values
            )
        if step == "STANDARDSCALER":
            df.iloc[:, :-N_META] = StandardScaler(
                with_mean=False, with_std=True
            ).fit_transform(df.iloc[:, :-N_META].values)

            # if "TEMPERATURE" not in step_slug and "HUMIDITY" not in step_slug and "PCA" not in step_slug:
            #     if "CWT" in step_slug:
            #         SampleVisualisation(df, CWT_Transform.shape, N_META, graph_outputdir + "/" + step, step_slug, None, None, CWT_Transform.scales)
            #
            #     if "STFT" in step_slug and "PCA" not in step_slug:
            #         SampleVisualisation(df, STFT_Transform.shape, N_META, graph_outputdir + "/" + step, step_slug,
            #                             STFT_Transform.sfft_window, STFT_Transform.stft_time, STFT_Transform.freqs)

        if step == "CENTER":
            df.iloc[:, :-N_META] = CenterScaler(center_by_sample=False).fit_transform(
                df.iloc[:, :-N_META].values
            )
        if step == "CENTER_STD":
            df.iloc[:, :-N_META] = CenterScaler(
                center_by_sample=True, divide_by_std=True
            ).fit_transform(df.iloc[:, :-N_META].values)
        if step == "MINMAX":
            df.iloc[:, :-N_META] = MinMaxScaler().fit_transform(
                df.iloc[:, :-N_META].values
            )
        if step == "ANSCOMBE":
            df.iloc[:, :-N_META] = Anscombe().transform(df.iloc[:, :-N_META].values)
        if step == "SQRT":
            df.iloc[:, :-N_META] = Sqrt().transform(df.iloc[:, :-N_META].values)
        if step == "LOG":
            df.iloc[:, :-N_META] = Log().transform(df.iloc[:, :-N_META].values)
        if step == "QN":
            df.iloc[:, :-N_META] = QuotientNormalizer(
                out_dir=graph_outputdir / step
            ).transform(df.iloc[:, :-N_META].values)
        if "STFT" in step:
            STFT_Transform = STFT(
                sfft_window=sfft_window,
                out_dir=graph_outputdir / step,
                step_slug=step_slug,
                animal_ids=animal_ids,
                targets=df["target"].tolist(),
                dates=df["date"].tolist(),
            )
            d = STFT_Transform.transform(df.copy().iloc[:, :-N_META].values)
            data_frame_stft = pd.DataFrame(d)
            data_frame_stft.index = df.index  # need to keep original sample index!!!!
            df_meta = df.iloc[:, -N_META:]
            df = pd.concat([data_frame_stft, df_meta], axis=1)
            del data_frame_stft
        if "CWT" in step:
            df_meta = df.iloc[:, -N_META:]
            df_o = df.copy()
            CWT_Transform = CWT(
                wavelet_f0=wavelet_f0,
                out_dir=graph_outputdir / step,
                step_slug=step_slug,
                n_scales=n_scales,
                animal_ids=animal_ids,
                targets=df["target"].tolist(),
                dates=df["date"].tolist(),
            )
            data_frame_cwt, data_frame_cwt_raw = CWT_Transform.transform(
                df.copy().iloc[:, :-N_META].values
            )
            data_frame_cwt = pd.DataFrame(data_frame_cwt)
            data_frame_cwt_raw = pd.DataFrame(data_frame_cwt_raw)

            # data_frame_cwt.index = df.index  # need to keep original sample index!!!!
            # df_meta = df.iloc[:, -N_META:]
            # df = pd.concat([data_frame_cwt, df_meta], axis=1)
            # sanity check#################################################################################################
            # wont work sincce using avg of sample!
            # rdm_idxs = random.choices(df.index.tolist(), k=1)
            # samples_tocheck = df_o.loc[(rdm_idxs), :].values[:, :-N_META]
            # cwt_to_check = pd.DataFrame(CWT(out_dir=graph_outputdir + "/" + step + "/cwt_sanity_check/").transform(samples_tocheck))
            # prev_cwt_results = df.loc[(rdm_idxs), :].values[:, :-N_META]
            # assert False not in (cwt_to_check.values == prev_cwt_results), "missmatch in cwt sample!"
            #############################################################################################################

            data_frame_cwt.index = df.index  # need to keep original sample index!!!!
            CWTVisualisation(
                step_slug,
                graph_outputdir,
                CWT_Transform.shape,
                CWT_Transform.coi_mask,
                CWT_Transform.scales,
                CWT_Transform.coi,
                df_o.copy(),
                data_frame_cwt,
                class_healthy_label,
                class_unhealthy_label,
                class_healthy_target,
                class_unhealthy_target,
            )

            data_frame_cwt_raw.index = (
                df.index
            )  # need to keep original sample index!!!!
            df = pd.concat([data_frame_cwt_raw, df_meta], axis=1)
            # CWTVisualisation(step_slug, graph_outputdir, CWT_Transform.shape, CWT_Transform.coi_mask, CWT_Transform.scales, CWT_Transform.coi, df_o.copy(),
            #                  data_frame_cwt_raw, class_healthy_label, class_unhealthy_label, class_healthy, class_unhealthy, filename_sub="real")

            df = df.dropna(axis=1, how="all")  # removes nan from coi
            del data_frame_cwt
            del data_frame_cwt_raw
        if "TSNE" in step:
            tsne_dim = int(step[step.find("(") + 1 : step.find(")")])
            print("tsne_dim", tsne_dim)
            df_before_reduction = df.iloc[:, :-N_META].values
            data_frame_tsne = pd.DataFrame(
                TSNE(n_components=tsne_dim).fit_transform(df_before_reduction)
            )
            data_frame_tsne.index = df.index  # need to keep original sample index!!!!
            df_meta = df.iloc[:, -N_META:]
            df = pd.concat([data_frame_tsne, df_meta], axis=1)
            del data_frame_tsne

        if "PCA" in step:
            pca_dim = int(step[step.find("(") + 1 : step.find(")")])
            print("pca_dim", pca_dim)
            df_before_reduction = df.iloc[:, :-N_META].values
            data_frame_pca = pd.DataFrame(
                PCA(n_components=pca_dim).fit_transform(df_before_reduction)
            )
            data_frame_pca.index = df.index  # need to keep original sample index!!!!
            df_meta = df.iloc[:, -N_META:]
            df = pd.concat([data_frame_pca, df_meta], axis=1)
            del data_frame_pca

        if "UMAP" in step:
            df_before_reduction = df.iloc[:, :-N_META].values
            data_frame_umap = pd.DataFrame(
                umap.UMAP().fit_transform(df_before_reduction)
            )
            data_frame_umap.index = df.index  # need to keep original sample index!!!!
            df_meta = df.iloc[:, -N_META:]
            df = pd.concat([data_frame_umap, df_meta], axis=1)
            del data_frame_umap

        print("AFTER STEP ->", df)
        if "CWT" not in step_slug:
            plotDistribution(
                df.iloc[:, :-N_META].values,
                graph_outputdir,
                f"data_distribution_after_{step}",
            )

    # if "PCA" in step_slug:
    #     plotDistribution(df.iloc[:, :-N_META].values, graph_outputdir, "data_distribution_after_%s" % step_slug)
    targets = df["target"]
    if keep_meta:
        df = df.iloc[:, :]
    else:
        df = df.iloc[:, :-N_META]
    df["target"] = targets
    print(df)
    return df


def main():
    print("")


if __name__ == "__main__":
    print("")

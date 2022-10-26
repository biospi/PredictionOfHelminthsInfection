import typer
from pathlib import Path
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import plotly
import h5py as h5
import pickle
from typing import List
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from preprocessing.preprocessing import apply_preprocessing_steps
from scipy.stats import entropy
from itertools import groupby

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


def load_herd(herdFile):
    herdData = {}

    ahf = h5.File(herdFile, "r")
    animal = list(ahf.keys())

    for i in animal:
        Tag = int(i)
        gdata = ahf[i]

        t = np.array(gdata["csTime"])
        x = np.array(gdata["cs"])
        cs = np.array([t, x])

        t = np.array(gdata["famachaTime"])
        x = np.array(gdata["famacha"])
        famacha = np.array([t, x])

        t = np.array(gdata["weightTime"])
        x = np.array(gdata["weight"])
        weight = np.array([t, x])
        herdData[Tag] = [Tag, famacha, cs, weight]
    ahf.close()
    return herdData


def concat_html(figs, filename):
    titles = []
    cpt = 1
    for x in range(0, len(figs)):
        titles.append(f"timeline")
        cpt += 1

    fig = make_subplots(
        subplot_titles=tuple(titles),
        rows=len(figs),
        cols=1,
        y_title="Activity",
        x_title="Time (1 min bins)",
    )

    for i, f in enumerate(figs):
        fig.append_trace(f[0], row=i + 1, col=1)
        fig.append_trace(f[1], row=i + 1, col=1)
        fig.append_trace(f[2], row=i + 1, col=1)
        fig.append_trace(f[3], row=i + 1, col=1)
        fig.append_trace(f[4], row=i + 1, col=1)
        fig.update_yaxes(type="log", row=i + 1, col=1)
    # fig.update_layout(height=200 * len(figs))
    fig.update(layout_showlegend=False)

    fig.write_html(filename)
    print(filename)


def sum_(to_resample):
    s = np.nan
    if to_resample.dropna().size > 0:
        s = np.sum(to_resample.dropna())
    return s


def split_given_size(a, size):
    return np.split(a, np.arange(size, len(a), size))


def predict_famacha(
    id,
    df,
    model_path,
    preprocessing_steps,
    output_dir,
    class_healthy_label,
    class_unhealthy_label,
    sample_size=10080,
    model_count=-1
):
    # first reshape the data as the same shape of the training samples
    chuncks = split_given_size(df["first_sensor_value"].values, sample_size)
    chuncks_li = chuncks
    try:
        chuncks = split_given_size(df["first_sensor_value_mrnn"].values, sample_size)
    except KeyError as e:
        print(e)

    chuncks_timestamp = split_given_size(df["timestamp"].values, sample_size)

    # for j in range(len(chuncks_timestamp)):
    #     print(df[df['timestamp'] == chuncks_timestamp[j][0]])

    samples = []

    idxs_to_rmv = []
    for i, s in enumerate(chuncks_li):
        m = np.nanmax(s)
        # plt.plot(s)
        # plt.title(f"max={m} n={i}")
        # plt.show()
        # print(i, m)
        if np.isnan(m) or m < 100:
            idxs_to_rmv.append(i)

        # if np.isnan(s).all() or np.all((s <= 1)) or np.all(s == s[0]): #to avoid testing on funky samples
        #     continue
        samples.append(s)

    data_frame = pd.DataFrame(samples)
    data_frame.replace([np.inf, -np.inf], np.nan, inplace=True)
    data_frame = data_frame.astype(np.float)
    data_frame = data_frame.fillna(1)

    data_frame["health"] = 0
    data_frame[
        "target"
    ] = 0  # add mock meta todo edit apply_processing_steps to handle no meta input

    models = list(model_path.glob("*.pkl"))
    if model_count > 0:
        models = models[0:model_count]

    # apply preprocessing
    m = ["health", "target"]
    if data_frame.shape[0] > 0:
        data_frame, _ = apply_preprocessing_steps(
            m,
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
            farm_name=f"{id}",
            keep_meta=True,
        )
        # if len(chuncks[0]) != sample_size:
        #     continue
        X_test = data_frame.iloc[:, :-len(m)]
        # y_pred_list = []
        # y_pred_proba_list = []

    for i, model_file in enumerate(models):
        print(f"model {i}/{len(models)} predicting X_test...")
        with open(str(model_file), "rb") as f:
            clf = pickle.load(f)
            y_pred = clf.predict(X_test.copy()).astype(float)
            y_pred_proba = clf.predict_proba(X_test.copy())[:, 1].astype(float)
            y_pred[idxs_to_rmv] = np.nan
            y_pred_proba[idxs_to_rmv] = np.nan
            # y_pred_list.append(y_pred)
            # y_pred_proba_list.append(y_pred_proba)
            df[f"famacha_pred_{i}"] = np.nan
            df[f"famacha_proba_{i}"] = np.nan

            if np.all(np.isnan(df["famacha"].values)):
                for n in range(X_test.shape[0]):
                    samp = X_test.iloc[n, :]
                    df[f"famacha_pred_{i}"].iloc[n * 1440 * 7] = y_pred[n]
                    df[f"famacha_proba_{i}"].iloc[n * 1440 * 7] = y_pred_proba[n]
                    # plt.plot(samp)
                    # plt.title(f"max={m} n={n}")
                    # plt.show()

                # cpt = 0
                # for n, v in enumerate(df["famacha"].values):
                #     if not np.isnan(v):
                #         # print(v)
                #         df[f"famacha_pred_{i}"].iloc[n] = y_pred[cpt]
                #         df[f"famacha_proba_{i}"].iloc[n] = y_pred_proba[cpt]
                #         cpt += 1

    return df, len(models)


def build_animal_pred(
    animal_file: Path = typer.Option(
        ..., exists=True, file_okay=True, dir_okay=False, resolve_path=True
    ),
    famacha_data: Path = typer.Option(
        ..., exists=True, file_okay=True, dir_okay=False, resolve_path=True
    ),
    model_path: Path = typer.Option(
        ..., exists=False, file_okay=False, dir_okay=True, resolve_path=True
    ),
    out_dir: Path = typer.Option(
        ..., exists=False, file_okay=False, dir_okay=True, resolve_path=True
    ),
    preprocessing_steps: List[str] = ["QN", "ANSCOMBE", "LOG"],
    class_healthy_label: List[str] = ["1To1"],
    class_unhealthy_label: List[str] = ["2To2"],
    res: str = "1D",
    model_count: int = -1
):
    id = animal_file.stem
    famacha_data = load_herd(famacha_data)
    i_list = []
    f_data = None
    for key, value in famacha_data.items():
        id_ = id[:-3] + f"{key}".zfill(3)
        i_list.append(id_)
        if id == id_:
            print(f"found famacha data for {id}")
            f_data = famacha_data[key]
            break
    print(i_list)

    if f_data is None:
        print(f"missing famacha data id={id}")
        f_data = [[[]], [[]], [[]], [[]]]

    df = pd.read_csv(animal_file)
    df["famacha"] = np.nan
    df["weight"] = np.nan
    df["cs"] = np.nan

    for i in range(len(f_data[1][0])):
        timestamp = f_data[1][0][i]
        d = df[df["timestamp"].isin([timestamp])]
        if d.shape[0] == 0:
            continue
        df.loc[d.index, "famacha"] = f_data[1][1][i]

    for i in range(len(f_data[2][0])):
        timestamp = f_data[2][0][i]
        d = df[df["timestamp"].isin([timestamp])]
        if d.shape[0] == 0:
            continue
        df.loc[d.index, "cs"] = f_data[2][1][i]

    for i in range(len(f_data[3][0])):
        timestamp = f_data[3][0][i]
        d = df[df["timestamp"].isin([timestamp])]
        if d.shape[0] == 0:
            continue
        df.loc[d.index, "weight"] = f_data[3][1][i]

    df, n_models = predict_famacha(
        id,
        df,
        model_path,
        preprocessing_steps,
        out_dir,
        class_healthy_label,
        class_unhealthy_label,
        model_count=model_count
    )
    # n = 1440 * 7 * 4 * 12 *2 # chunk row size
    # list_df = [df[i : i + n] for i in range(0, df.shape[0], n)]
    # print(f"found {len(list_df)} weeks.")
    # figs = []
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # for i, df in enumerate(list_df):
    df.index = pd.to_datetime(df.date_str)
    # df_resampled = df.resample(res).sum()
    agg_dict = {
        "timestamp": "first",
        "date_str": "first",
        "first_sensor_value": "sum",
        "famacha": "first",
        "weight": "first",
        "cs": "first",
    }

    for n in range(n_models):
        agg_dict[f"famacha_pred_{n}"] = "first"
        agg_dict[f"famacha_proba_{n}"] = "first"

    df_resampled = df.resample(res).agg(agg_dict, skipna=False)

    activity = df_resampled["first_sensor_value"].values
    weight = df_resampled["weight"].values
    # if len(weight[weight > 0]) > 0:
    #     weight[:] = weight[weight > 0][0]

    cs = df_resampled["cs"].values
    famacha = df_resampled["famacha"].values

    famacha_predicted_dec = np.nanmean(
        df_resampled.loc[:, df_resampled.columns.str.startswith("famacha_pred")].values,
        axis=1,
    )

    famacha_predicted_proba = np.nanmean(
        df_resampled.loc[
            :, df_resampled.columns.str.startswith("famacha_proba")
        ].values,
        axis=1,
    )

    famacha_predicted = (famacha_predicted_dec > 0.5).astype(float)
    famacha_predicted[np.isnan(famacha_predicted_dec)] = np.nan

    # if len(famacha[famacha > 0]) > 0:
    #     famacha[:] = famacha[famacha > 0][0]

    time_axis = df_resampled.index

    trace_a = go.Line(
        x=time_axis, y=activity, name=f"Activity ({res} bin)", marker_color="steelblue"
    )
    fig.add_trace(trace_a, secondary_y=False)

    data_f_inc = famacha.copy()
    famacha_inc = famacha[~np.isnan(famacha)] - np.roll(famacha[~np.isnan(famacha)], 1)
    famacha_inc = (famacha_inc >= 1).astype(int)
    cpt = 0
    for i in range(len(data_f_inc)):
        if np.isnan(data_f_inc[i]):
            continue
        data_f_inc[i] = famacha_inc[cpt]
        cpt += 1

    if not np.all(np.isnan(famacha)):
        trace_f = go.Scatter(
            x=time_axis,
            y=famacha,
            opacity=0.8,
            line_color="black",
            name="famacha score (real)",
            mode="lines+markers",
            # marker_color=[DEFAULT_PLOTLY_COLORS[int(x)] if not np.isnan(x) else np.nan for x in famacha],
            marker={"symbol": "x", "size": 7},
            connectgaps=True,
        )
        fig.add_trace(trace_f, secondary_y=True)

    trace_f_inc = go.Scatter(
        x=time_axis,
        y=data_f_inc,
        opacity=1,
        line_color="black",
        name="famacha score increase(real)",
        mode="lines+markers",
        # marker_color=[DEFAULT_PLOTLY_COLORS[int(x)] if not np.isnan(x) else np.nan for x in famacha],
        marker={"symbol": "x", "size": 7},
        connectgaps=True,
    )
    fig.add_trace(trace_f_inc, secondary_y=True)

    pred_correct = []
    cpt_c = 0
    cpt_ic = 0
    for y, pred in zip(famacha, famacha_predicted):
        if y == 1 and pred == 0:  # famacha is 1 and pred is healthy
            pred_correct.append("green")
            cpt_c += 1
            continue
        if y > 1 and pred == 1:  # famacha is >1 and pred is unhealthy
            pred_correct.append("green")
            cpt_c += 1
            continue
        if np.isnan(y):
            pred_correct.append(np.nan)
            continue

        pred_correct.append("red")
        cpt_ic += 1

    y_true = (famacha[~np.isnan(famacha)] > 1).astype(int)
    y_pred = famacha_predicted[~np.isnan(famacha_predicted)]
    y_proba = famacha_predicted_proba[~np.isnan(famacha_predicted_proba)]
    tnr = 0
    fpr = 0
    fpr = 0
    fnr = 0
    tpr = 0
    if len(y_true) > 0 and len(y_true) == len(y_pred):
        tnr, fpr, fnr, tpr = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        tnr = tnr / sum(y_true == 0)
        fpr = fpr / sum(y_true == 1)
        fnr = fnr / sum(y_true == 0)
        tpr = tpr / sum(y_true == 1)

    # trace_f_pred = go.Scatter(
    #     x=time_axis,
    #     y=famacha_predicted,
    #     opacity=0.8,
    #     line_color="gray",
    #     name="famacha score increase (predicted)",
    #     mode="lines+markers",
    #     marker_color=pred_correct,
    #     marker={"symbol": "circle-open", "size": 15},
    #     connectgaps=True,
    # )
    # fig.add_trace(trace_f_pred, secondary_y=True)

    trace_f_pred_dec = go.Scatter(
        x=time_axis,
        y=famacha_predicted_dec,
        opacity=0.8,
        line_color="blue",
        name=f"mean (n_models={n_models}) famacha score increase (predicted)",
        mode="lines+markers",
        marker={"symbol": "x", "size": 7},
        connectgaps=True,
    )
    fig.add_trace(trace_f_pred_dec, secondary_y=True)

    trace_f_proba_pred_dec = go.Scatter(
        x=time_axis,
        y=famacha_predicted_proba,
        opacity=0.8,
        line_color="purple",
        name=f"mean (n_models={n_models}) famacha score increase probability (predicted)",
        mode="lines+markers",
        marker={"symbol": "x", "size": 7},
        connectgaps=True,
    )
    fig.add_trace(trace_f_proba_pred_dec, secondary_y=True)

    # trace_c = go.Scatter(
    #     x=time_axis,
    #     y=cs,
    #     opacity=0.9,
    #     mode="lines+markers",
    #     name="condition score",
    #     connectgaps=True,
    # )
    # fig.add_trace(trace_c, secondary_y=True)
    #
    # trace_w = go.Scatter(
    #     x=time_axis,
    #     y=weight,
    #     opacity=0.9,
    #     name="weight",
    #     marker_color="black",
    #     mode="lines+markers",
    #     connectgaps=True,
    # )
    # fig.add_trace(trace_w, secondary_y=True)

    fig.update_layout(
        title_text=f"Timeline of transponder {id} | TPR={tpr:.2f} FPR={fpr:.2f} TNR={tnr:.2f} FNR={fnr:.2f} | CORRECT={cpt_c} INCORRECT={cpt_ic}"
    )
    fig.update_yaxes(title_text="<b>Activity</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>Meta Data</b>", secondary_y=True)

    # figs.append([trace_a, trace_w, trace_f, trace_f_pred, trace_c])

    filename = f"{int(tpr * 100):03}_{id}.html"
    out_dir.mkdir(parents=True, exist_ok=True)
    filepath = str(out_dir / filename)
    fig.write_html(filepath)
    print(filepath)
    # concat_html(figs, filepath)
    return (
        out_dir,
        n_models,
        time_axis,
        res,
        activity,
        famacha,
        famacha_predicted_dec,
        famacha_predicted_proba,
    )


def build_herd_pred(
    out_dir,
    n_models,
    time_axis,
    res,
    herd,
    famacha_list,
    famacha_predicted_dec_list,
    famacha_predicted_proba_list,
):
    # print(herd, famacha_list, famacha_predicted_dec_list)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    herd = pd.DataFrame(herd).values
    activity_herd = np.mean(herd, axis=0)
    famacha_list = pd.DataFrame(famacha_list).values
    famacha = np.nanmean(famacha_list, axis=0)
    famacha_predicted_dec_list = pd.DataFrame(famacha_predicted_dec_list).values
    famacha_pred = np.nanmean(famacha_predicted_dec_list, axis=0)
    famacha_predicted_proba_list = pd.DataFrame(famacha_predicted_proba_list).values
    famacha_proba_pred = np.nanmean(famacha_predicted_proba_list, axis=0)

    trace_a = go.Line(
        x=time_axis,
        y=activity_herd,
        name=f"Herd activity ({res} bin)",
        marker_color="steelblue",
    )
    fig.add_trace(trace_a, secondary_y=False)

    if not np.all(np.isnan(famacha)):
        trace_f = go.Scatter(
            x=time_axis,
            y=famacha,
            opacity=0.8,
            line_color="black",
            name="herd famacha score (real)",
            mode="lines+markers",
            # marker_color=[DEFAULT_PLOTLY_COLORS[int(x)] if not np.isnan(x) else np.nan for x in famacha],
            marker={"symbol": "x", "size": 7},
            connectgaps=True,
        )
        fig.add_trace(trace_f, secondary_y=True)

    data_f_inc = famacha.copy()
    famacha_inc = famacha[~np.isnan(famacha)] - np.roll(famacha[~np.isnan(famacha)], 1)
    famacha_inc = (famacha_inc > 0).astype(int)
    print(famacha_inc)
    cpt = 0
    for i in range(len(data_f_inc)):
        if np.isnan(data_f_inc[i]):
            continue
        data_f_inc[i] = famacha_inc[cpt]
        cpt += 1

    trace_f_inc = go.Scatter(
        x=time_axis,
        y=data_f_inc,
        opacity=1,
        line_color="black",
        name="famacha score increase(real)",
        mode="lines+markers",
        # marker_color=[DEFAULT_PLOTLY_COLORS[int(x)] if not np.isnan(x) else np.nan for x in famacha],
        marker={"symbol": "x", "size": 7},
        connectgaps=True,
    )
    fig.add_trace(trace_f_inc, secondary_y=True)

    trace_f_proba_pred_dec = go.Scatter(
        x=time_axis,
        y=famacha_proba_pred,
        opacity=0.8,
        line_color="purple",
        name=f"mean (n_models={n_models}) herd famacha score increase probability (predicted)",
        mode="lines+markers",
        marker={"symbol": "x", "size": 7},
        connectgaps=True,
    )
    fig.add_trace(trace_f_proba_pred_dec, secondary_y=True)

    trace_f_pred_dec = go.Scatter(
        x=time_axis,
        y=famacha_pred,
        opacity=0.8,
        line_color="blue",
        name=f"mean (n_models={n_models}) herd famacha score increase (predicted)",
        mode="lines+markers",
        marker={"symbol": "x", "size": 7},
        connectgaps=True,
    )
    fig.add_trace(trace_f_pred_dec, secondary_y=True)

    trace_f_pred_binary = go.Scatter(
        x=time_axis,
        y=(famacha_pred > 0.5).astype(int),
        opacity=0.8,
        line_color="firebrick",
        name=f"binary (n_models={n_models}) herd famacha score increase (predicted)",
        mode="lines+markers",
        marker=dict(opacity=0),
        connectgaps=True,
    )
    fig.add_trace(trace_f_pred_binary, secondary_y=True)

    cpt = 0
    for item in famacha_list:
        if np.all(np.isnan(item)):
            cpt += 1

    y_true = (famacha[famacha > 0]).astype(int)
    y_pred = (famacha_pred[famacha > 0] > 0.5).astype(int)
    y_proba = famacha_proba_pred[famacha > 0]
    tnr = 0
    fpr = 0
    fpr = 0
    fnr = 0
    tpr = 0
    if len(y_true) > 0:
        tnr, fpr, fnr, tpr = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        tnr = tnr / sum(y_true == 0)
        fpr = fpr / sum(y_true == 1)
        fnr = fnr / sum(y_true == 0)
        tpr = tpr / sum(y_true == 1)

    pred_correct = []
    cpt_c = 0
    cpt_ic = 0
    for y, pred in zip(famacha, famacha_pred):
        if y == 1 and pred == 0:  # famacha is 1 and pred is healthy
            pred_correct.append("green")
            cpt_c += 1
            continue
        if y > 1 and pred == 1:  # famacha is >1 and pred is unhealthy
            pred_correct.append("green")
            cpt_c += 1
            continue
        if np.isnan(y):
            pred_correct.append(np.nan)
            continue

        pred_correct.append("red")
        cpt_ic += 1

    fig.update_layout(
        title_text=f"Timeline of herd(total={len(herd)}), with famacha {len(herd) - cpt}, without famacha {cpt}| TPR={tpr:.2f} FPR={fpr:.2f} TNR={tnr:.2f} FNR={fnr:.2f} | CORRECT={cpt_c} INCORRECT={cpt_ic}"
    )

    fig.update_yaxes(title_text="<b>Activity</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>Meta Data</b>", secondary_y=True)
    filename = f"herd.html"
    out_dir.mkdir(parents=True, exist_ok=True)
    filepath = str(out_dir / filename)
    fig.write_html(filepath)
    print(filepath)


def main(activity_files=None, famacha_h5=None, models=None, out=None, model_count=-1):
    herd = []
    famacha_list = []
    famacha_predicted_dec_list = []
    famacha_predicted_proba_list = []
    for i, activity_file in enumerate(activity_files.glob("*.csv")):
        (
            out_dir,
            n_models,
            time_axis,
            res,
            activity,
            famacha,
            famacha_predicted_dec,
            famacha_predicted_proba,
        ) = build_animal_pred(
            activity_file,
            famacha_h5,
            models,
            out,
            model_count=model_count
        )

        herd.append(activity)
        famacha_list.append(famacha)
        famacha_predicted_dec_list.append(famacha_predicted_dec)
        famacha_predicted_proba_list.append(famacha_predicted_proba)

    build_herd_pred(
        out_dir,
        n_models,
        time_axis,
        res,
        herd,
        famacha_list,
        famacha_predicted_dec_list,
        famacha_predicted_proba_list,
    )


if __name__ == "__main__":
    main(
        activity_files=Path(
            "E:/thesis/activity_data/cedara/6_missingrate_[0.0]_seql_1440_iteration_100_hw__n_591"
        ),
        famacha_h5=Path("F:/Data2/cedara_animal_data.h5"),
        models=Path(
            "E:/thesis2/main_experiment/cedara_RepeatedKFold_7_7_QN_ANSCOMBE_LOG_season_False/2To2/models/SVC_linear_7_QN_ANSCOMBE_LOG"
        ),
        model_count=50,
        out=Path("E:/thesis/timelines/cedara"),
    )
    main(
        activity_files=Path(
            "F:/MRNN/imputed_data/4_missingrate_[0.0]_seql_1440_iteration_100_hw__n_421"
        ),
        famacha_h5=Path("F:/Data2/delmas_animal_data.h5"),
        models=Path(
            "E:/thesis2/main_experiment/delmas_RepeatedKFold_7_7_QN_ANSCOMBE_LOG_season_False/2To2/models/SVC_linear_7_QN_ANSCOMBE_LOG"
        ),
        model_count=50,
        out=Path("E:/thesis/timelines/delmas"),
    )
    # typer.run(main)

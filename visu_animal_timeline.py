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

from preprocessing.preprocessing import apply_preprocessing_steps

DEFAULT_PLOTLY_COLORS=['rgb(31, 119, 180)', 'rgb(255, 127, 14)',
                       'rgb(44, 160, 44)', 'rgb(214, 39, 40)',
                       'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
                       'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
                       'rgb(188, 189, 34)', 'rgb(23, 190, 207)']


def load_herd(herdFile):
    herdData = {}

    ahf = h5.File(herdFile, 'r')
    animal = list(ahf.keys())

    for i in animal:
        Tag = int(i)
        gdata = ahf[i]

        t = np.array(gdata['csTime'])
        x = np.array(gdata['cs'])
        cs = np.array([t, x])

        t = np.array(gdata['famachaTime'])
        x = np.array(gdata['famacha'])
        famacha = np.array([t, x])

        t = np.array(gdata['weightTime'])
        x = np.array(gdata['weight'])
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
    #fig.update_layout(height=200 * len(figs))
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


def predict_famacha(id, df, model_path, preprocessing_steps, output_dir, class_healthy_label, class_unhealthy_label, sample_size=10080):
    models = list(model_path.glob('*.pkl'))
    for i, model_file in enumerate(models):
        with open(str(model_file), 'rb') as f:
            clf = pickle.load(f)
            chuncks = split_given_size(df["first_sensor_value_mrnn"].values, sample_size)
            samples = []
            for s in chuncks:
                samples.append(s)
            data_frame = pd.DataFrame(samples)
            data_frame.replace([np.inf, -np.inf], np.nan, inplace=True)
            data_frame = data_frame.fillna(1)
            data_frame = data_frame.astype(np.float)
            data_frame["health"] = 0
            data_frame["target"] = 0 #add mock meta todo edit apply_processing_steps to handle no meta input
            data_frame, _ = apply_preprocessing_steps(
                ["meta"],
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
                keep_meta=False,
            )
                # if len(chuncks[0]) != sample_size:
                #     continue
            X_test = data_frame.iloc[:, :-2].values
            #X_test = X_test.reshape(-1, 1)
            y_pred = clf.predict(X_test.copy())

            df["famacha_pred"] = np.nan
            cpt = 0
            for n, v in enumerate(df["famacha"].values):
                if not np.isnan(v):
                    #print(v)
                    df["famacha_pred"].iloc[n] = y_pred[cpt]
                    cpt += 1

    return df


def main(
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
):
    id = animal_file.stem
    famacha_data = load_herd(famacha_data)
    i_list = []
    f_data = None
    for key, value in famacha_data.items():
        id_ = id[:-3]+f"{key}".zfill(3)
        i_list.append(id_)
        if id == id_:
            print(f"found famacha data for {id}")
            f_data = famacha_data[key]
            break
    print(i_list)

    if f_data is None:
        print("missing famacha data")
        return

    df = pd.read_csv(animal_file)
    df["famacha"] = np.nan
    df["weight"] = np.nan
    df["cs"] = np.nan

    for i in range(len(f_data[1][0])):
        timestamp = f_data[1][0][i]
        d = df[df['timestamp'].isin([timestamp])]
        if d.shape[0] == 0:
            continue
        df.loc[d.index, "famacha"] = f_data[1][1][i]

    for i in range(len(f_data[2][0])):
        timestamp = f_data[2][0][i]
        d = df[df['timestamp'].isin([timestamp])]
        if d.shape[0] == 0:
            continue
        df.loc[d.index, "cs"] = f_data[2][1][i]

    for i in range(len(f_data[3][0])):
        timestamp = f_data[3][0][i]
        d = df[df['timestamp'].isin([timestamp])]
        if d.shape[0] == 0:
            continue
        df.loc[d.index, "weight"] = f_data[3][1][i]

    df = predict_famacha(id, df, model_path, preprocessing_steps, out_dir, class_healthy_label, class_unhealthy_label)
    #n = 1440 * 7 * 4 * 12 *2 # chunk row size
    #list_df = [df[i : i + n] for i in range(0, df.shape[0], n)]
    #print(f"found {len(list_df)} weeks.")
    #figs = []
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    #for i, df in enumerate(list_df):
    df.index = pd.to_datetime(df.date_str)
    #df_resampled = df.resample(res).sum()
    df_resampled = df.resample(res).agg(
        # dict(
        #     timestamp="first",
        #     date_str="first",
        #     first_sensor_value="sum",
        #     signal_strength="mean",
        #     battery_voltage="mean",
        #     xmin="sum",
        #     xmax="sum",
        #     ymin="sum",
        #     ymax="sum",
        #     zmin="sum",
        #     zmax="sum",
        #     famacha="first",
        #     weight="first",
        #     cs="first",
        # ),
        dict(
            timestamp="first",
            date_str="first",
            first_sensor_value="sum",
            famacha="first",
            famacha_pred="first",
            weight="first",
            cs="first",
        ),
        skipna=False,
    )
    activity = df_resampled["first_sensor_value"].values
    weight = df_resampled["weight"].values
    # if len(weight[weight > 0]) > 0:
    #     weight[:] = weight[weight > 0][0]

    cs = df_resampled["cs"].values
    famacha = df_resampled["famacha"].values
    famacha_predicted = df_resampled["famacha_pred"].values

    # if len(famacha[famacha > 0]) > 0:
    #     famacha[:] = famacha[famacha > 0][0]

    time_axis = df_resampled.index

    trace_a = go.Line(
        x=time_axis,
        y=activity,
        name=f"Activity ({res} bin)",
        marker_color='steelblue'
    )
    fig.add_trace(trace_a, secondary_y=False)

    trace_w = go.Scatter(
        x=time_axis,
        y=weight,
        opacity=.9,
        name="weight",
        marker_color="black",
        mode='lines+markers',
        connectgaps=True
    )
    fig.add_trace(trace_w, secondary_y=True)

    # c = "green"
    # if famacha[0] == 2:
    #     c = "orange"
    # if famacha[0] > 2:
    #     c = "red"

    trace_f = go.Scatter(
        x=time_axis,
        y=famacha,
        opacity=.8,
        name="famacha score (real)",
        mode='lines+markers',
        marker_color=[DEFAULT_PLOTLY_COLORS[int(x)] if not np.isnan(x) else np.nan for x in famacha],
        marker={
            'symbol': 'x',
            'size': 7
        },
        connectgaps=True
    )
    fig.add_trace(trace_f, secondary_y=True)

    trace_f_pred = go.Scatter(
        x=time_axis,
        y=famacha_predicted,
        opacity=.8,
        name="famacha score (predicted)",
        mode='lines+markers',
        marker_color=[DEFAULT_PLOTLY_COLORS[int(x)] if not np.isnan(x) else np.nan for x in famacha],
        marker={
            'symbol': 'circle',
            'size': 15
        },
        connectgaps=True
    )
    fig.add_trace(trace_f_pred, secondary_y=True)

    trace_c = go.Scatter(
        x=time_axis,
        y=cs,
        opacity=.9,
        mode='lines+markers',
        name="condition score",
        connectgaps=True
    )
    fig.add_trace(trace_c, secondary_y=True)

    fig.update_layout(title_text=f'Timeline of transponder {id}')
    fig.update_yaxes(title_text="<b>Activity</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>Meta Data</b>", secondary_y=True)

    #figs.append([trace_a, trace_w, trace_f, trace_f_pred, trace_c])

    filename = f"{int(np.nansum(activity))}_{int(np.nansum(famacha))}_{id}.html"
    out_dir.mkdir(parents=True, exist_ok=True)
    filepath = str(out_dir / filename)
    fig.write_html(filepath)
    print(filepath)
    #concat_html(figs, filepath)


def local_run():
    for activity_file in Path("F:/MRNN/imputed_data/4_missingrate_[0.0]_seql_1440_iteration_100_hw__n_421").glob("*.csv"):
        main(activity_file, Path("F:/Data2/delmas_animal_data.h5"),
             Path("E:/thesis2/main_experiment/delmas_RepeatedKFold_7_7_QN_ANSCOMBE_LOG_season_False/2To2/models/SVC_linear_7_QN_ANSCOMBE_LOG"),
             Path("E:/thesis2/timelines"))


if __name__ == "__main__":
    local_run()
    #typer.run(main)

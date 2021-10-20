import typer
from pathlib import Path
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import plotly
import h5py as h5

DEFAULT_PLOTLY_COLORS=['rgb(31, 119, 180)', 'rgb(255, 127, 14)',
                       'rgb(44, 160, 44)', 'rgb(214, 39, 40)',
                       'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
                       'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
                       'rgb(188, 189, 34)', 'rgb(23, 190, 207)']


def loadHerd(herdFile):
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
        titles.append(f"Week {cpt} data")
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
        fig.update_yaxes(type="log", row=i + 1, col=1)
    fig.update_layout(height=200 * len(figs))
    fig.update(layout_showlegend=False)

    fig.write_html(filename)
    print(filename)


def sum_(to_resample):
    s = np.nan
    if to_resample.dropna().size > 0:
        s = np.sum(to_resample.dropna())
    return s


def main(
    animal_file: Path = typer.Option(
        ..., exists=True, file_okay=True, dir_okay=False, resolve_path=True
    ),
    famacha_data: Path = typer.Option(
        ..., exists=True, file_okay=True, dir_okay=False, resolve_path=True
    ),
    out_dir: Path = typer.Option(
        ..., exists=False, file_okay=False, dir_okay=True, resolve_path=True
    ),
    res: str = "10T",
):
    id = animal_file.stem
    famacha_data = loadHerd(famacha_data)
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

    assert f_data is not None, "missing famacha data"

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

    n = 1440 * 7  # chunk row size
    list_df = [df[i : i + n] for i in range(0, df.shape[0], n)]
    print(f"found {len(list_df)} weeks.")

    figs = []
    for i, df in enumerate(list_df):
        df.index = pd.to_datetime(df.date_str)
        #df_resampled = df.resample(res).sum()
        df_resampled = df.resample(res).agg(
            dict(
                timestamp="first",
                date_str="first",
                first_sensor_value="sum",
                signal_strength="mean",
                battery_voltage="mean",
                xmin="sum",
                xmax="sum",
                ymin="sum",
                ymax="sum",
                zmin="sum",
                zmax="sum",
                famacha="first",
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

        if len(famacha[famacha > 0]) > 0:
            famacha[:] = famacha[famacha > 0][0]

        time_axis = df_resampled.index

        trace_a = go.Bar(
            x=time_axis,
            y=activity,
            marker_color='steelblue'
        )

        trace_w = go.Scatter(
            x=time_axis,
            y=weight,
            mode='lines+markers',
            marker_symbol='x',
            marker_color="black"
        )

        c = "green"
        if famacha[0] == 2:
            c = "orange"
        if famacha[0] > 2:
            c = "red"

        trace_f = go.Scatter(
            x=time_axis,
            y=famacha,
            mode='lines',
            marker_color=c
        )

        trace_c = go.Scatter(
            x=time_axis,
            y=cs,
            mode='lines+markers',
            marker_color="blue"
        )

        figs.append([trace_a, trace_w, trace_f, trace_c])

    filename_concat = f"{id}.html"
    out_dir.mkdir(parents=True, exist_ok=True)
    concat_html(figs, str(out_dir / filename_concat))


if __name__ == "__main__":
    typer.run(main)

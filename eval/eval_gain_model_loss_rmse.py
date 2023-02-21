from pathlib import Path
import ast
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Manager, Pool

import pandas as pd

marker = ['x', 's', 'o', 'D', 'P', '^', '.', '>', '|']
import matplotlib.colors as mcolors

def missingness_robustness_plot(data_dir, df, study_id="delmas", li=False):
    iteration = 99
    n_top_traces = 20
    df_ = df[df["n_top_traces"] == n_top_traces]
    df_ = df_[df_["i"] == iteration]
    df_ = df_[df_["missing_rate"] <= 0.8]
    name = "Sample length\n(in days)"
    df_ = df_.rename(columns={"seq_len": f"{name}"})
    metric = "rmse"
    if li:
        metric = "rmse_li"
    ax = df_.pivot("missing_rate", name, metric).plot(
        kind="line",
        linestyle='--',
        figsize=(3, 7),
        rot=0,
        grid=True,
        title=f"Evolution of RMSE\nwith increasing missingness\n(Model iteration={iteration+1})",
    )
    ax.set_xticks(df_["missing_rate"].unique())
    for n in sorted(df_[name].unique()):
        dat = df_[df_[name] == n].sort_values(name)
        intervals = pd.eval(dat[f"{metric}_boot"])
        perct = np.percentile(intervals, [2.5, 50, 97.5], axis=1)
        top = perct[2, :]
        bottom = perct[0, :]
        x = dat["missing_rate"].values
        ax.fill_between(x, top.astype(float), bottom.astype(float), alpha=0.2)
    a = []
    for i, line in enumerate(ax.get_lines()):
        line.set_marker(marker[i])
        a.append(line.get_label())
    ax.legend(ax.get_lines(), a, loc='best', title=ax.get_legend().get_title().get_text())
    ax.set(xlabel="Missing rate (in percent)", ylabel="RMSE")
    filename = f"{study_id}_{metric}_missingness_gain.png"
    filepath = data_dir / filename
    print(filepath)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.show()


def window_length_effect_plot(data_dir, df, study_id="delmas", li=False):
    iteration = 99
    n_top_traces = 20
    df_ = df[df["n_top_traces"] == n_top_traces]
    df_ = df_[df_["i"] == iteration]
    df_ = df_[df_["missing_rate"] <= 0.8]
    name = "Sample length\n(in days)"
    df_ = df_.rename(columns={"seq_len": f"{name}"})
    metric = "rmse"
    if li:
        metric = "rmse_li"
    ax = df_.pivot(name, "missing_rate", metric).plot(
        figsize=(3, 7),
        kind="line",
        linestyle='--',
        rot=0,
        grid=True,
        title=f"Evolution of RMSE\nwith increasing sample length\n(Model iteration={iteration+1})",
    )
    ax.set_xticks(sorted(df_[name].unique()))
    for n in df_["missing_rate"].unique():
        dat = df_[df_["missing_rate"] == n].sort_values(name)
        intervals = pd.eval(dat[f"{metric}_boot"])
        perct = np.percentile(intervals, [2.5, 50, 97.5], axis=1)
        top = perct[2, :]
        bottom = perct[0, :]
        x = dat[name].values
        ax.fill_between(x, top.astype(float), bottom.astype(float), alpha=0.2)
    a = []
    for i, line in enumerate(ax.get_lines()):
        line.set_marker(marker[i])
        a.append(line.get_label())
    ax.legend(ax.get_lines(), a, loc='best', title=ax.get_legend().get_title().get_text())
    ax.set(xlabel="Sample length (in days)", ylabel="RMSE")
    filename = f"{study_id}_{metric}_seqlen_gain.png"
    filepath = data_dir / filename
    print(filepath)
    plt.tight_layout()
    plt.savefig(filepath)


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def loss_curve(data_dir, df, study_id="delmas", li=False):
    mvavrg_n = 10
    fig, ax = plt.subplots(figsize=(9, 7))
    df = df.sort_values(by=["seq_len", "i"])
    df_n = df[df["n_top_traces"] == 40]
    colors = list(mcolors.TABLEAU_COLORS.keys())
    for s_l in df_n["seq_len"].unique():
        df_ = df_n[df_n["seq_len"] == s_l]
        #df_ = df_[df_["i"] > 1]
        iterations = df_["i"].values[::15]
        g_loss = df_["g_loss"].values[::15]
        d_loss = df_["d_loss"].values[::15]
        g_loss = moving_average(g_loss, mvavrg_n)
        d_loss = moving_average(d_loss, mvavrg_n)
        ax.plot(iterations[0:len(g_loss)], g_loss, label=f'Generator loss ({s_l} {"day" if s_l == 1 else "days"})', marker='x', linestyle='--', c=colors[s_l])
        ax.plot(iterations[0:len(d_loss)], d_loss, label=f'Discriminator loss ({s_l} {"day" if s_l == 1 else "days"})', marker='o', c=colors[s_l], alpha=0.85)
    ax.set_ylabel(f'Loss (moving average on {mvavrg_n} points)')
    ax.set_xlabel('Iterations')
    ax.set_title('GAIN models loss for different sample length'.title())
    #ax.grid()
    ax.legend()
    fig.tight_layout()
    filename = f"{study_id}_loss_curve_gain.png"
    filepath = data_dir / filename
    print(filepath)
    fig.savefig(filepath)
    plt.show()


def n_transponder_effect_plot(data_dir, df, study_id="delmas"):
    iteration = 99
    missing_rate = 0.1
    seq_len = 1
    df_ = df[df["seq_len"] == seq_len]
    df_ = df_[df_["i"] == iteration]
    #df_ = df_.sort_values(by="n_top_traces")
    # name = "Sample length\n(in days)"
    # df_ = df_.rename(columns={"seq_len": f"{name}"})
    df_["n_top_traces"] = df["n_top_traces"].astype(int)
    ax = df_.pivot("n_top_traces", "missing_rate", "rmse").plot(
        kind="line",
        linestyle='--',
        rot=0,
        grid=True,
        title=f"Evolution of RMSE with number of transponders\n(Model iteration={iteration} Missing rate={missing_rate})",
    )
    a = []
    for i, line in enumerate(ax.get_lines()):
        line.set_marker(marker[i])
        a.append(line.get_label())
    ax.legend(ax.get_lines(), a, loc='best', title=ax.get_legend().get_title().get_text())

    ax.set(xlabel="Number of transponders", ylabel="RMSE")
    filename = f"{study_id}_rmse_ntransponder_gain.png"
    filepath = data_dir / filename
    print(filepath)
    plt.savefig(filepath)


def worker(dfs, i, tot, file):
    print(f"{i}/{tot} {file}...")
    df = pd.read_csv(file)
    df["seq_len"] = int(ast.literal_eval(df["training_shape"][0])[1] / 1440)
    missing_rate = int(file.parent.stem.split("_")[3]) / 100
    df["missing_rate"] = missing_rate
    df["n_top_traces"] = int(file.parent.stem.split("_")[-1])
    dfs.append(df)


if __name__ == "__main__":
    data_dir = Path("H:/fo18103/gain/gain/delmas_exp")
    files = [x for x in data_dir.rglob("*.csv")]
    files = [x for x in files if "rmse" in x.stem.lower()]
    # dfs = []
    # for i, file in enumerate(files):
    #     print(f"{i}/{len(files)} {file}...")
    #     df = pd.read_csv(file)
    #     df["seq_len"] = int(ast.literal_eval(df["training_shape"][0])[1] / 1440)
    #     missing_rate = int(file.parent.stem.split("_")[3]) / 100
    #     df["missing_rate"] = missing_rate
    #     df["n_top_traces"] = int(file.parent.stem.split("_")[-1])
    #     dfs.append(df)

    pool = Pool(processes=6)
    with Manager() as manager:
        dfs = manager.list()
        for i, file in enumerate(files):
            pool.apply_async(
                worker,
                (dfs, i, len(files), file),
            )
        pool.close()
        pool.join()
        pool.terminate()

        dfs = list(dfs)

    df = pd.concat(dfs)
    df = df.sort_values(by="missing_rate")
    df = df.reset_index(drop=True)
    loss_curve(data_dir, df, study_id="delmas")
    n_transponder_effect_plot(data_dir, df, study_id="delmas")
    missingness_robustness_plot(data_dir, df, study_id="delmas", li=True)
    window_length_effect_plot(data_dir, df, study_id="delmas", li=True)
    missingness_robustness_plot(data_dir, df, study_id="delmas")
    window_length_effect_plot(data_dir, df, study_id="delmas")

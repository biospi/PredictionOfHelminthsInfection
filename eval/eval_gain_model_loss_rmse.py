from pathlib import Path
import ast
import matplotlib.pyplot as plt

import pandas as pd


def missingness_robustness_plot(data_dir, df, study_id="delmas"):
    iteration = 10
    n_top_traces = 40
    df_ = df[df["n_top_traces"] == n_top_traces]
    df_ = df_[df_["i"] == iteration]
    name = "Sample length\n(in days)"
    df_ = df_.rename(columns={"seq_len": f"{name}"})
    ax = df_.pivot("missing_rate", name, "rmse").plot(
        kind="bar",
        rot=0,
        title=f"Evolution of RMSE with increasing missingness\n(Model iteration={iteration})",
    )
    ax.set(xlabel="Missing rate (in percent)", ylabel="RMSE")
    filename = f"{study_id}_rmse_missingness_gain.png"
    filepath = data_dir / filename
    print(filepath)
    plt.savefig(filepath)


def window_length_effect_plot(data_dir, df, study_id="delmas"):
    iteration = 99
    n_top_traces = 40
    df_ = df[df["n_top_traces"] == n_top_traces]
    df_ = df_[df_["i"] == iteration]
    name = "Sample length\n(in days)"
    df_ = df_.rename(columns={"seq_len": f"{name}"})
    ax = df_.pivot(name, "missing_rate", "rmse").plot(
        kind="bar",
        rot=0,
        title=f"Evolution of RMSE with increasing sample length\n(Model iteration={iteration})",
    )
    ax.set(xlabel="Sample length (in days)", ylabel="RMSE")
    filename = f"{study_id}_rmse_seqlen_gain.png"
    filepath = data_dir / filename
    print(filepath)
    plt.savefig(filepath)


def n_transponder_effect_plot(data_dir, df, study_id="delmas"):
    iteration = 99
    #missing_rate = 0.1
    seq_len = 1
    df_ = df[df["seq_len"] == seq_len]
    df_ = df_[df_["i"] == iteration]

    #df_ = df_.sort_values(by="n_top_traces")
    # name = "Sample length\n(in days)"
    # df_ = df_.rename(columns={"seq_len": f"{name}"})
    ax = df_.pivot("n_top_traces", "missing_rate", "rmse").plot(
        kind="bar",
        rot=0,
        title=f"Evolution of RMSE with number of transponders\n(Model iteration={iteration} Missing rate={missing_rate})",
    )
    ax.set(xlabel="Sample length (in days)", ylabel="RMSE")
    filename = f"{study_id}_rmse_ntransponder_gain.png"
    filepath = data_dir / filename
    print(filepath)
    plt.savefig(filepath)


if __name__ == "__main__":
    data_dir = Path("H:/fo18103/gain/delmas")
    files = [x for x in data_dir.rglob("*.csv")]
    files = [x for x in files if "rmse" in x.stem.lower()]
    dfs = []
    for i, file in enumerate(files):
        print(f"{i}/{len(files)} {file}...")
        df = pd.read_csv(file)
        df["seq_len"] = int(ast.literal_eval(df["training_shape"][0])[1] / 1440)
        missing_rate = int(file.parent.stem.split("_")[3]) / 100
        df["missing_rate"] = missing_rate
        df["n_top_traces"] = int(file.parent.stem.split("_")[-1])
        dfs.append(df)

    df = pd.concat(dfs)
    df = df.sort_values(by="missing_rate")
    df = df.reset_index(drop=True)
    n_transponder_effect_plot(data_dir, df, study_id="delmas")
    missingness_robustness_plot(data_dir, df, study_id="delmas")
    window_length_effect_plot(data_dir, df, study_id="delmas")

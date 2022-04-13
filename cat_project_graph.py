from pathlib import Path, PurePath
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import typer
from tqdm import tqdm
import matplotlib.cm as cm

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


def mean_confidence_interval(x):
    # boot_median = [np.median(np.random.choice(x, len(x))) for _ in range(iteration)]
    x.values.sort()
    lo_x_boot = np.percentile(x, 2.5)
    hi_x_boot = np.percentile(x, 97.5)
    # print(lo_x_boot, hi_x_boot)
    return lo_x_boot, hi_x_boot


def main(
    input_dir: Path = typer.Option(
        ..., exists=True, file_okay=False, dir_okay=True, resolve_path=True
    ),
    out_dir: Path = typer.Option(
        ..., exists=True, file_okay=False, dir_okay=True, resolve_path=True
    ),
):
    data = []
    for p in input_dir.rglob("*.json"):
        if "proba" in str(p):
            continue
        split = list(PurePath(p).parts)
        data.append(split + [p])

    df = pd.DataFrame(data)
    dfs = [group for _, group in df.groupby(df[df.shape[1]-3])]
    print(f"there are {len(dfs)} different processing pipeline.")
    fig, ax1 = plt.subplots(figsize=(20.48, 11.52))
    ax2 = ax1.twinx()
    for i, df in tqdm(enumerate(dfs), total=len(dfs)):
        data_xaxis = []
        data_yaxis = []
        auc_list = []
        n_samples = []
        for index, row in df.iterrows():
            res_file_path = row[df.shape[1]-1]
            p_steps = row[df.shape[1]-3]
            thresh = row[df.shape[1]-5]
            t_v = thresh.replace("_", ".")
            thresh_float = float(t_v)
            results = json.load(open(res_file_path))
            clf_res = results[list(results.keys())[0]]
            aucs = []
            for item in clf_res:
                if "auc" not in item:
                    continue
                auc = item["auc"]
                aucs.append(auc)
                training_shape = item["training_shape"][0]
                testing_shape = item["testing_shape"][0]
            auc_list.append(aucs)
            median_auc = np.median(aucs)
            data_xaxis.append(thresh_float)
            data_yaxis.append(median_auc)
            n_samples.append(training_shape+testing_shape)

        out_dir.mkdir(parents=True, exist_ok=True)
        df_ = pd.DataFrame(auc_list)
        data_yaxis = np.array(data_yaxis)
        df_ci = pd.DataFrame(df_.apply(mean_confidence_interval, axis=1).tolist(), index=df_.index)

        ax1.plot(
            data_xaxis,
            data_yaxis,
            label=f"Median AUC | processing steps={p_steps}",
            marker="x"
        )

        ax2.plot(
            data_xaxis,
            n_samples,
            marker="x",
            color="grey"
        )

        ax1.fill_between(data_xaxis, df_ci[0].values, df_ci[1].values, alpha=0.2)
        ax1.legend()
        ax1.set_xticks(data_xaxis, rotation=-45)
        ax2.set_xticks(data_xaxis, rotation=-45)
        ax2.set_yticks(n_samples)

    ax1.axhline(y=0.5, color='black', linestyle='--')
    ax1.xaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    fig.suptitle("Intensity threshold and AUC (CI=0.5% 99% percentile)")
    ax1.set_xlabel("Intensity threshold")
    ax1.set_ylabel("Mean AUC")
    ax2.set_ylabel("Number of samples(high activity peak window)")
    ax1.set_xticklabels(data_xaxis, rotation=-45)
    ax2.set_xticklabels(data_xaxis, rotation=-45)
    #plt.legend()
    ax2.legend().set_visible(False)
    filename = f"auc_intensity.png"
    filepath = out_dir / filename
    print(filepath)
    fig.savefig(filepath)


if __name__ == "__main__":
    typer.run(main)

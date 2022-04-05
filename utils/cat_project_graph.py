from pathlib import Path
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


def main():
    input_dir = Path("E:/Cats/ml_peak_build_sec_w4min")
    out_dir = Path("E:/Cats/ml_peak_build_sec_w4min")
    data = []
    for p in input_dir.rglob("*.json"):
        if 'proba' in str(p):
            continue
        data.append(str(p).split('\\')+[p])

    df = pd.DataFrame(data)
    dfs = [group for _, group in df.groupby(df[5])]
    print(f"there are {len(dfs)} different processing pipeline.")
    fig, ax = plt.subplots(figsize=(19.80, 7.20))
    for i, df in enumerate(dfs):
        print(f"{i}/{len(dfs)}...")
        data_xaxis = []
        data_yaxis = []
        for index, row in df.iterrows():
            res_file_path = row[7]
            p_steps = row[5]
            thresh = row[3]
            thresh_float = float(thresh.replace('_', '.'))
            results = json.load(open(res_file_path))
            clf_res = results[list(results.keys())[0]]
            aucs = []
            for item in clf_res:
                if "auc" not in item:
                    continue
                auc = item["auc"]
                aucs.append(auc)
            mean_auc = np.mean(aucs)
            data_xaxis.append(thresh_float)
            data_yaxis.append(mean_auc)

        out_dir.mkdir(parents=True, exist_ok=True)
        plt.plot(data_xaxis, data_yaxis, label=f"processing steps={p_steps}", marker="x")
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    fig.suptitle('Intensity threshold and AUC')
    plt.xlabel('Intensity threshold')
    plt.ylabel('Mean AUC')
    plt.legend()
    filename = f"auc_intensity.png"
    filepath = out_dir / filename
    print(filepath)
    fig.savefig(filepath)


if __name__ == "__main__":
    print([x.stem for x in Path("E:/Cats/build_sec2/peak").glob("*")])
    main()


import pandas as pd
from pathlib import Path

if __name__ == "__main__":
    # farm = "delmas"
    # filepath = "E:/Data2/debug3/delmas/dataset4_mrnn_7day/activity_farmid_dbft_7_1min.csv"

    farm = "cedara"
    filepath = "E:/Data2/debug3/cedara/dataset6_mrnn_7day/activity_farmid_dbft_7_1min.csv"

    data_frame = pd.read_csv(filepath, sep=",", header=None, low_memory=False)
    meta_columns = [
        "label",
        "id",
        "imputed_days",
        "date"
    ]

    data_frame = data_frame.astype(dtype=float, errors='ignore')  # cast numeric values as float
    data_point_count = data_frame.shape[1]
    hearder = [str(n) for n in range(0, data_point_count)]

    for i, m in enumerate(meta_columns[::-1]):
        hearder[-i-1] = m
    data_frame.columns = hearder
    data_frame = data_frame.dropna()
    data_frame[data_frame.columns.values[:-len(meta_columns)]] = data_frame[
        data_frame.columns.values[:-len(meta_columns)]].clip(lower=0)

    df_activity_window = data_frame.iloc[:, data_frame.columns.str.isnumeric()]
    df_meta = data_frame[meta_columns]

    for a in [7, 6, 5, 4, 3, 2, 1]:
        df_activity_window = df_activity_window.iloc[:, -(1440*a):]
        data_frame = pd.concat([df_activity_window, df_meta], axis=1)
        out_dir = Path(f"E:/Data2/wfmm/{farm}")
        filename = f"{a}.csv"
        out_dir.mkdir(parents=True, exist_ok=True)
        filepath = out_dir / filename
        print(filepath)
        data_frame.to_csv(filepath, index=False)

    print(data_frame)
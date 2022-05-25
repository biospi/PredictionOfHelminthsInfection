import pandas as pd
import numpy as np
from pathlib import Path


def main(file=Path("E:/thesis/datasets/cedara/dataset6_mrnn_7day/activity_farmid_dbft_7_1min.csv")):
    print(file)
    df = pd.read_csv(file, header=None)
    print(df)
    dates = pd.to_datetime(df.iloc[:, -1])
    mask = dates > pd.Timestamp(2013, 2, 14)
    df_clipped = df[mask]
    filename = file.parent / f"{file.stem}_clipped.csv"
    print(filename)
    df_clipped.to_csv(filename, sep=",", index=False, header=False)


if __name__ == "__main__":
    main()
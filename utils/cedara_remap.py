from pathlib import Path
import pandas as pd
import numpy as np

if __name__ == "__main__":
    path = "F:/Data2/backfill_1min/cedara_70091100056/"
    dataDir = Path(path)
    files = sorted(dataDir.glob("*.csv"))
    names = np.array([x.stem[-3:] for x in files]).astype(int)
    files = [[x, int(x.stem)] for x in files]

    print(files)

    mapping_file = "F:/Data2/2012 nommers vd bokke_print.csv"

    df = pd.read_csv(mapping_file, sep=",")

    A = df["Transponder number"].values
    B = names.astype(int)

    intersect = np.intersect1d(np.hstack(A), B)

    df_intersect = df[df["Transponder number"].isin(intersect)]

    for file in files:

        old_id = int(str(file[1])[-4:])
        new_id = df[df["Transponder number"] == old_id]["Transponder number 14Feb2013"].values
        if new_id.size == 0:
            continue
        new_id = int(new_id[0])
        new_filename = str(file[0].absolute()).replace(str(old_id), str(new_id))
        print("before=", str(file[0].absolute()))
        print("after=", str(new_filename))


    print(df)
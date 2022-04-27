from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from tqdm import tqdm


def local_run():
    main(data_dir=Path("E:/thesis/activity_data/cedara/backfill_1min_cedara_fixed_with_missing_tag"))
    main(data_dir=Path("F:/Data2/backfill_1min_delmas_fixed"))


def main(data_dir: Path = typer.Option(
            ..., exists=False, file_okay=False, dir_okay=True, resolve_path=True
        ),
):
    print(data_dir)
    files = sorted(data_dir.glob('*.csv'))
    print(files)
    timestamp_list = []
    for i, file in enumerate(tqdm(files)):
        df = pd.read_csv(file)
        timestamp = pd.to_datetime(df["date_str"]).astype('M8[m]').astype('O').values
        timestamp_list.append(timestamp[0])
        timestamp_list.append(timestamp[-1])
    timestamp_list = np.array(timestamp_list).flatten()
    start_date = np.min(timestamp_list)
    end_date = np.max(timestamp_list)
    print(f"min timestamp={start_date} max timestamp={end_date}")
    minutes = []
    n_minutes_in_between = int(np.ceil((end_date - start_date).total_seconds() / 60))
    for i in range(n_minutes_in_between):
        m = start_date + timedelta(minutes=i)
        minutes.append(m)
    #add extra min
    m = start_date + timedelta(minutes=i+1)
    minutes.append(m)

    minutes = np.array(minutes)
    df_ = pd.DataFrame(minutes, columns=["datetime"])
    df_["first_sensor_value"] = np.nan
    df_["date_str"] = df_['datetime'].dt.strftime('%Y-%m-%dT%H:%M')
    df_["timestamp"] = df_['datetime'].astype(np.int64) // 10 ** 9
    df_ = df_.drop('datetime', axis=1)

    filepath = data_dir / f"{len(file.stem) * '9'}.csv"
    print(filepath)
    df_.to_csv(filepath)


if __name__ == '__main__':
    local_run()
    #typer.run(main)
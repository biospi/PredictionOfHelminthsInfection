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
    for i, file in enumerate(tqdm(files)):
        df = pd.read_csv(file)
        break
    df_ = df.copy()
    df_["first_sensor_value"] = np.nan

    filepath = data_dir / f"{len(file.stem) * '9'}.csv"
    print(filepath)
    df_.to_csv(filepath, index=False)


if __name__ == '__main__':
    local_run()
    #typer.run(main)
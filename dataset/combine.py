import typer
from pathlib import Path
import pandas as pd
import numpy as np

from model.data_loader import parse_param_from_filename, load_activity_data


def group_by_animals(dataframe):
    list_of_df = [g for _, g in dataframe.groupby(['id'])]
    return list_of_df


def get_consec_week_mask(diff):
    mask = (diff == 7).astype(int)
    mask2 = np.zeros(mask.shape)
    for i in range(mask.size - 1):
        if mask[i] == 1 and mask[i + 1] == 1:
            mask2[i] = 1
            mask2[i+1] = 1

    return mask2 > 0


def combine(data):
    for df in group_by_animals(data):
        dates = pd.to_datetime(df['date'], dayfirst=True).values
        d1 = dates[:-1]
        d2 = dates[1:]
        diff = (d2 - d1) / np.timedelta64(1, 'D')
        diff = np.append(diff, np.nan)
        mask = get_consec_week_mask(diff)
        if df[mask].shape[0] == 0:
            continue
        print(df[mask])
        print(diff)
        print(mask)


def main(
    dataset: Path = typer.Option(
        ..., exists=False, file_okay=True, dir_okay=False, resolve_path=True
    ),
    class_healthy_label: str = "1To1",
    class_unhealthy_label: str = "2To2",
):
    print(dataset)
    """This script combines consecutive samples(pair activity, famacha)\n
    Args:\n
        dataset: Dataset file
    """
    file = str(dataset)

    days, farm_id, option, sampling = parse_param_from_filename(file)
    print(f"loading dataset file {file} ...")
    (
        data_frame,
        N_META,
        class_healthy_target,
        class_unhealthy_target,
        label_series,
        samples
    ) = load_activity_data(file, days, class_healthy_label, class_unhealthy_label)

    #print(data_frame)

    combine(data_frame)


if __name__ == "__main__":
    typer.run(main)

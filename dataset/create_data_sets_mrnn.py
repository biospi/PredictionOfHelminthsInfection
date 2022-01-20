import typer
from pathlib import Path
import pandas as pd


def main(
    data_file: Path = typer.Option(
        ..., exists=False, file_okay=True, dir_okay=False, resolve_path=True
    ),
    herd_file: Path = typer.Option(
            ..., exists=False, file_okay=True, dir_okay=False, resolve_path=True
    )
):
    df = pd.read_csv(data_file, header=None)
    hearder = [str(n) for n in df.columns.values]
    N_META = 4
    hearder[-4] = 'label'
    hearder[-3] = 'id'
    hearder[-2] = 'imputed_days'
    hearder[-1] = 'date'
    df.columns = hearder

    df_herd = pd.read_csv(herd_file)
    df_herd['date_datetime'] = df_herd['date_str']
    print(df)
    for index, row in df.iterrows():
        activity = row[:-N_META]
        date = pd.to_datetime(row['date'], format="%d/%m/%Y").strftime('%Y-%m-%dT%H:%M')
        date_range = pd.date_range(end=date, periods=10081, freq='T').strftime('%Y-%m-%dT%H:%M')
        mrnn_window = df_herd[df_herd['date_datetime'].isin(date_range)]
        filename = f"sample_{str(index).zfill(6)}_{row['date'].replace('/','-')}_{row['id']}_{row['label']}_mrnn_window.csv"
        out_dir = data_file.parent / "mrnn_windows"
        out_dir.mkdir(parents=True, exist_ok=True)
        filepath = out_dir / filename
        print(filepath)
        mrnn_window.to_csv(filepath, index=False)


if __name__ == "__main__":
    #typer.run(main)
    main(data_file=Path("E:/Data2/debug3/delmas/datasetraw_none_7day/activity_farmid_dbft_7_1min.csv"),
         herd_file=Path('C:/Users/fo18103/PycharmProjects/PredictionOfHelminthsInfection/Data/delmas_activity_data_weather.csv'))

#
# Author: Ranjeet Bhamber <ranjeet <a.t> bristol.ac.uk>
#
# Copyright (C) 2020  Biospi Laboratory for Medical Bioinformatics, University of Bristol, UK
#
# This file is part of PredictionOfHelminthsInfection.
#
# PHI is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PHI is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with seaMass.  If not, see <http://www.gnu.org/licenses/>.
#

# %%

import typer

from dataset.herd import *
from dataset.samples import *
from dataset.cmdsextra import bc
import datetime
import json

#--fam-file F:\Data2\delmas_animal_data.h5 --data-dir "F:\MRNN\imputed_data\1_missingrate_[0.0]_seql_10080_iteration_100_hw__n_298" --out-dir "E:\Data2\debug\delmas\dataset_mrnn_7day"
#--fam-file F:\Data2\cedara_animal_data.h5 --data-dir "F:\MRNN\imputed_data\3_missingrate_[0.0]_seql_10080_iteration_100_hw__n_238" --out-dir "E:\Data2\cedara\delmas\dataset_mrnn_7day"


def local_run():

    main(fam_file=Path("F:/Data2/delmas_animal_data.h5"),
         data_dir=Path("F:/MRNN/imputed_data_/8_missingrate_[0.0]_seql_1440_iteration_1000_hw__n_316"),
         out_dir=Path("E:/thesis/datasets/delmas/delmas_dataset_mrnn_30days"),
         data_col="first_sensor_value_mrnn",
         n_days=7,
         remove_missing=False)

    # main(fam_file=Path("F:/Data2/cedara_animal_data.h5"),
    #      data_dir=Path("F:/MRNN/imputed_data/4_4000_missingrate_0_seql_1440_iteration_100_hw__n_325"),
    #      out_dir=Path("E:/thesis/datasets/cedara/cedara_dataset_mrnn_30days"),
    #      data_col="first_sensor_value_mrnn",
    #      n_days=30,
    #      remove_missing=False)


    # main(fam_file=Path("F:/Data2/delmas_animal_data.h5"),
    #      data_dir=Path("E:/thesis/gain/delmas/1_miss_rate_0000_iteration_0100_thresh_-1_anscombe_False_n_top_traces_66"),
    #      out_dir=Path("E:/thesis/datasets/delmas/delmas_dataset_1_gain_66_no_filter"),
    #      data_col="first_sensor_value_gain",
    #      n_days=7,
    #      remove_missing=False)

    # main(fam_file=Path("F:/Data2/delmas_animal_data.h5"),
    #      data_dir=Path("E:/thesis/gain/delmas/1_miss_rate_0000_iteration_0100_thresh_-1_anscombe_False_n_top_traces_66"),
    #      out_dir=Path("E:/thesis/datasets/delmas/delmas_dataset_1_li_66_no_filter"),
    #      data_col="first_sensor_value_li",
    #      n_days=7,
    #      remove_missing=False)

    # main(fam_file=Path("F:/Data2/cedara_animal_data.h5"),
    #      data_dir=Path("E:/thesis/gain/cedara/1_miss_rate_0000_iteration_0100_thresh_-1_anscombe_False_n_top_traces_172"),
    #      out_dir=Path("E:/thesis/datasets/cedara/cedara_dataset_1_gain_172_no_filter"),
    #      data_col="first_sensor_value_gain",
    #      n_days=7,
    #      remove_missing=False)
    #
    # main(fam_file=Path("F:/Data2/cedara_animal_data.h5"),
    #      data_dir=Path("E:/thesis/gain/cedara/1_miss_rate_0000_iteration_0100_thresh_-1_anscombe_False_n_top_traces_172"),
    #      out_dir=Path("E:/thesis/datasets/cedara/cedara_dataset_1_li_172_no_filter"),
    #      data_col="first_sensor_value_li",
    #      n_days=7,
    #      remove_missing=False)

    # main(fam_file=Path("F:/Data2/delmas_animal_data.h5"),
    #      data_dir=Path("H:/fo18103/gain/gain/delmas/1_miss_rate_0000_iteration_0100_thresh_100_anscombe_False_n_top_traces_66"),
    #      out_dir=Path("E:/thesis/datasets/delmas/delmas_dataset_1_gain_66"),
    #      data_col="first_sensor_value_gain",
    #      n_days=7,
    #      remove_missing=False)
    #
    # main(fam_file=Path("F:/Data2/delmas_animal_data.h5"),
    #      data_dir=Path("H:/fo18103/gain/gain/delmas/1_miss_rate_0000_iteration_0100_thresh_100_anscombe_False_n_top_traces_66"),
    #      out_dir=Path("E:/thesis/datasets/delmas/delmas_dataset_1_li_66"),
    #      data_col="first_sensor_value_li",
    #      n_days=7,
    #      remove_missing=False)
    #
    # main(fam_file=Path("F:/Data2/cedara_animal_data.h5"),
    #      data_dir=Path("H:/fo18103/gain/gain/cedara/1_miss_rate_0000_iteration_0100_thresh_100_anscombe_False_n_top_traces_172"),
    #      out_dir=Path("E:/thesis/datasets/cedara/cedara_dataset_1_gain_172"),
    #      data_col="first_sensor_value_gain",
    #      n_days=7,
    #      remove_missing=False)
    #
    # main(fam_file=Path("F:/Data2/cedara_animal_data.h5"),
    #      data_dir=Path("H:/fo18103/gain/gain/cedara/1_miss_rate_0000_iteration_0100_thresh_100_anscombe_False_n_top_traces_172"),
    #      out_dir=Path("E:/thesis/datasets/cedara/cedara_dataset_1_gain_li_172"),
    #      data_col="first_sensor_value_li",
    #      n_days=7,
    #      remove_missing=False)


    # main(fam_file=Path("F:/Data2/cedara_animal_data.h5"),
    #      data_dir=Path("F:/MRNN/imputed_data/23_missingrate_[0.0]_seql_1440_iteration_100_hw__n_325"),
    #      out_dir=Path("E:/thesis/datasets/cedara/dataset_li_7_23"),
    #      data_col="first_sensor_value_li",
    #      n_days=7,
    #      remove_missing=False)
    #
    # main(fam_file=Path("F:/Data2/delmas_animal_data.h5"),
    #      data_dir=Path("F:/MRNN/imputed_data/17_missingrate_[0.0]_seql_1440_iteration_100_hw__n_299"),
    #      out_dir=Path("E:/thesis/datasets/delmas/dataset_li_7_17"),
    #      data_col="first_sensor_value_li",
    #      n_days=7,
    #      remove_missing=False)


    # main(fam_file=Path("F:/Data2/delmas_animal_data.h5"),
    #      data_dir=Path("E:/thesis/gain/delmas/miss_rate_0000_iteration_1000_thresh_50_anscombe_False_n_top_traces_60"),
    #      out_dir=Path("E:/thesis/datasets/delmas/datasetmrnn7_gain_a"),
    #      data_col="first_sensor_value_gain",
    #      n_days=7,
    #      remove_missing=False)
    #
    # main(fam_file=Path("F:/Data2/cedara_animal_data.h5"),
    #      data_dir=Path("E:/thesis/gain/cedara/miss_rate_0000_iteration_1000_thresh_50_anscombe_False_n_top_traces_60"),
    #      out_dir=Path("E:/thesis/datasets/cedara/datasetmrnn7_gain_a"),
    #      data_col="first_sensor_value_gain",
    #      n_days=7,
    #      remove_missing=False)

    # main(fam_file=Path("F:/Data2/cedara_animal_data.h5"),
    #      data_dir=Path("E:/thesis/activity_data/cedara/backfill_1min_cedara_fixed_with_missing_tag"),
    #      out_dir=Path("E:/thesis_debug/dataset/cedara"), remove_missing=False)


def main(
        fam_file: Path = typer.Option(
            ..., exists=False, file_okay=True, dir_okay=False, resolve_path=True
        ),
        data_dir: Path = typer.Option(
            ..., exists=False, file_okay=False, dir_okay=True, resolve_path=True
        ),
        out_dir: Path = typer.Option(
            ..., exists=False, file_okay=False, dir_okay=True, resolve_path=True
        ),
        data_col: str = "first_sensor_value_mrnn",
        n_days: int = 7,
        farm_id: str = "farmid",
        night: bool = False,
        remove_missing: bool = False
):
    """This script builds activity/ground truth datasets\n
    Args:\n
        famFile: Famacha HDF5 file
        dataDir: Directory with activity data
        outDir: Output Directory
        data_col: Name of data column in imputed file (first_sensor_value_gain | first_sensor_value | first_sensor_value_li)
        ndays: Number of days in samples
        night: Use night data
    """

    print("Loading Famacha and Sorce based data from HDF5 file")
    file_herd = herd_file(fam_file)
    famachaHerd = herd_data()

    print("Loading Activity traces and Times of famacha based animals")
    activity_data = activity_file(data_dir, data_col)

    # load Famacha based scores from HDF5
    file_herd.load_herd(famachaHerd)

    # load Filenames of activity data.
    sheepID = activity_data.getID()

    famachaHerd.set_id_of_herd(sheepID)

    mis = famachaHerd.get_missing_list()

    if remove_missing:
        print("remove_missing")
        famachaHerd.remove_missing()

    # Load only data based on Famacha data.
    samples = SampleSet()

    samples.generateSet(famachaHerd, activity_data, n_days)

    # aTraces = activityData.loadActivityTraceList(famachaHerd.getAnimalIDList())

    # sTraces = activityData.loadActivityTrace(famachaHerd.herd[0].ID)

    # totalS = len([ele for sub in samples.set for ele in sub])
    # validS = len([ele for sub in samples.set for ele in sub if ele.valid == True])
    # falseS = len([ele for sub in samples.set for ele in sub if ele.valid == False])

    totalS = len(samples.set)
    validS = len([x for x in samples.set if x.valid == True])
    falseS = len([x for x in samples.set if x.valid == False])

    print(f"{bc.MAG}Summary of extracted samples:{bc.ENDC}")

    print(f"Number of Samples extracted: {bc.BLUE}{totalS} {bc.ENDC}")
    print(f"Number of Valid Samples extracted: {bc.GREEN}{validS} {bc.ENDC}")
    print(f"Number of NaN Samples extracted: {bc.RED}{falseS} {bc.ENDC}")

    # Get all meta data for each sample im our generated data set

    metaSet = np.array(samples.getSet())
    actSet = samples.getActivity()
    timeSet = samples.getiTime()

    set1To2 = samples.getFamachaCase("1To2")
    idx = samples.getFamachaCase("1To2")
    set1To2M = metaSet[idx]
    set1To2A = actSet[idx]
    set1To2T = timeSet[idx]

    # targets_info = {"total": {"all": totalS}}
    targets_info = {
        "total": {
            "all": totalS,
            "valid": np.where(samples.valid == True)[0].size,
            "not_valid": np.where(samples.valid == False)[0].size,
        }
    }
    for target in list(set(samples.df)):
        targets_info[target] = {
            "all": np.where(samples.df == target)[0].size,
            "valid": 0,
            "not_valid": 0,
        }

    # targets_info = {"total": {"valid": np.where(samples.valid == True)[0].size, "not_valid": np.where(samples.valid == False)[0].size, "all": totalS}}
    # for target in list(set(samples.df)):
    #     targets_info[target] = {"total": np.where(samples.df == target)[0].size, "valid": 0, "not_valid": 0}

    for i in np.where(samples.valid == False)[0]:
        for k in targets_info.keys():
            if samples.df[i] == k:
                targets_info[k]["not_valid"] += 1

    for i in np.where(samples.valid == True)[0]:
        for k in targets_info.keys():
            if samples.df[i] == k:
                targets_info[k]["valid"] += 1
    #
    # for k in targets_info.keys():
    #     targets_info[k]["all"] = targets_info[k]["not_valid"] + targets_info[k]["valid"]

    split = data_dir.name.split("_")

    # farm = "delmas"
    # base_station = 70101200027

    # farm = "cedara"
    # base_station = 70091100056
    out_dir.mkdir(parents=True, exist_ok=True)
    # farm_id = str(dataDir).split('\\')[-1]

    filename = out_dir / f"activity_{farm_id}_dbft_{n_days}_1min.json"
    print(filename)
    json.dump(targets_info, open(str(filename), "w"))

    s = []
    # meta = [0, '02/12/2015', '01/12/2015', 40101310050, 2, 1, 1, 1, -1, '02/12/2015', '15/12/2015', '08/01/2016',
    # '15/01/2016', '22/01/2016', 13, 24, 31, 7]
    # valid_idx = np.where(samples.valid == True)[0]
    for idx in range(totalS):
        # meta[1] = datetime.datetime.fromtimestamp(samples.iT[idx][-1]).strftime('%d/%m/%Y')
        # meta[2] = datetime.datetime.fromtimestamp(samples.iT[idx][0]).strftime('%d/%m/%Y')
        # meta[9] = datetime.datetime.fromtimestamp(samples.iT[idx][-1]).strftime('%d/%m/%Y')
        # meta[3] = samples.set[idx].ID
        dayTime = np.array(
            [
                "Day" if 6 <= x.hour <= 18 else "Night"
                for x in pd.to_datetime(samples.iT[idx], unit="s")
            ]
        )
        if night:
            sample_activity = samples.iA[idx][dayTime == "Night"].tolist()
        else:
            sample_activity = samples.iA[idx].tolist()
        sample = (
                sample_activity
                + [samples.df[idx]]
                + [
                    samples.set[idx].ID,
                    samples.set[idx].missRate,
                    datetime.datetime.fromtimestamp(samples.iT[idx][-1]).strftime(
                        "%d/%m/%Y"
                    ),
                ]
        )
        s.append(sample)
    df = pd.DataFrame(s)
    df.to_csv(str(filename).replace(".json", ".csv"), sep=",", index=False, header=False)
    print(filename)
    del samples
    return str(out_dir)


if __name__ == "__main__":
    local_run()
    #typer.run(main)

#
# Author: Axel Montout <axel.montout <a.t> bristol.ac.uk>
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

import argparse
import glob
import os
from pathlib import Path

import imputation
from commands import createDataSets
from pipeline import add_exogenous
import ml
import numpy as np

def imputed_data_exists(path):
    folders = [x[0] for x in os.walk(path) if len(x) > 0 and "miss_rate" in x[0]]
    return len(folders) > 0, folders[0] if len(folders) > 0 else []


def famacha_dataset_exists(path):
    folders = [x[0] for x in os.walk(path) if len(x) > 0 and "dataset" in x[0] and "temp" not in x[0]and "hum" not in x[0] and "temp_humidity" not in x[0]]
    return len(folders) > 0, [x for x in folders if "gain" in x][::-1], [x for x in folders if "li" in x][::-1]


def weather_files_exist(path):
    temperature_files = [glob.glob(x[0] + "/*.csv")[0] for x in os.walk(path) if len(x) > 0 and "temp" in x[0] and "temp_humidity" not in x[0]]
    humidity_files = [glob.glob(x[0] + "/*.csv")[0] for x in os.walk(path) if len(x) > 0 and "hum" in x[0] and "temp_humidity" not in x[0]]
    return len(temperature_files) > 0 and len(humidity_files) > 0, temperature_files[::-1], humidity_files[::-1]


if __name__ == "__main__":
    print("starting job...")
    print("imputation...")

    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('fam_file', help='Famacha HDF5 file', type=str)
    parser.add_argument('weather_file', help='Weather file', type=str)

    parser.add_argument('--batch_size', help='for imputation, the number of samples in mini-batch', default=128,
                        type=int)
    parser.add_argument('--hint_rate', help='for imputation, hint probability', default=0.9, type=float)
    parser.add_argument('--alpha', help='for imputation, hyperparameter', default=100, type=float)
    parser.add_argument('--iterations', help='for imputation, number of training interations', default=600, type=int)
    parser.add_argument('--miss_rate', help='for imputation, missing data probability', default=0.0, type=float)
    parser.add_argument('--n_job', type=int, default=6, help='Number of thread to use.')
    parser.add_argument('--n_top_traces', type=int, default=-1,
                        help='for imputation, select n traces with highest entropy (<= 0 to select all traces)')
    parser.add_argument('--enable_anscombe', help="for imputation, appy anscombe on activity count before imputation",
                        type=bool, default=False)
    parser.add_argument('--enable_remove_zeros',
                        help="for imputation, remove zero counts in activity before imputation", type=bool,
                        default=False)
    parser.add_argument('--enable_log_anscombe',
                        help="for imputation, appy log(anscombe) on activity count before imputation", type=bool,
                        default=True)
    parser.add_argument('--window', type=bool, default=False)
    parser.add_argument('--export_csv', type=bool, help="for imputation, export imputed traces as csv", default=True)
    parser.add_argument('--export_traces', type=bool, default=True)
    parser.add_argument('--reshape', type=str, help="for imputation, reshape activity traces to 1 day chunck",
                        default='y')
    parser.add_argument('--w', type=str, default='n')
    parser.add_argument('--add_t_col', help="for imputation, add time column in reshape", type=str, default='y')
    parser.add_argument('--thresh_daytime', help="for imputation, minimum number of positive count in 1 day.",
                        default=50, type=int)
    parser.add_argument('--thresh_nan_ratio', help="for imputation, max percent of nan allowed in 1 day.", default=90,
                        type=int)

    parser.add_argument('--ndays', help="Number of days in samples", default=7, type=int)

    args = parser.parse_args()

    exist, i_path = imputed_data_exists(args.output_dir)
    if not exist:
        imputed_data_dir = imputation.start(args)
    else:
        print("imputation folder exists! Skipping to dataset creation...")
        imputed_data_dir = i_path
    # imputed_data_dir = "F:\Data2\job_debug\miss_rate_0_0_iteration_0600_thresh_100_anscombe_False_n_top_traces_17"
    print("imputation done.")

    dataset_exist, dataset_files_gain, dataset_files_li = famacha_dataset_exists(args.output_dir)

    if not dataset_exist:
        famFile = Path(args.fam_file)
        dataDir = Path(imputed_data_dir)
        dataset_files_gain = []
        dataset_files_li = []
        dataset_files_gain.append(
            createDataSets.main(famFile, dataDir, Path(args.output_dir + "/dataset_gain_7day"), "first_sensor_value_gain",
                                7))
        dataset_files_gain.append(
            createDataSets.main(famFile, dataDir, Path(args.output_dir + "/dataset_gain_6day"), "first_sensor_value_gain",
                                6))
        dataset_files_gain.append(
            createDataSets.main(famFile, dataDir, Path(args.output_dir + "/dataset_gain_5day"), "first_sensor_value_gain",
                                5))
        dataset_files_gain.append(
            createDataSets.main(famFile, dataDir, Path(args.output_dir + "/dataset_gain_4day"), "first_sensor_value_gain",
                                4))
        dataset_files_gain.append(
            createDataSets.main(famFile, dataDir, Path(args.output_dir + "/dataset_gain_3day"), "first_sensor_value_gain",
                                3))
        dataset_files_gain.append(
            createDataSets.main(famFile, dataDir, Path(args.output_dir + "/dataset_gain_2day"), "first_sensor_value_gain",
                                2))
        dataset_files_gain.append(
            createDataSets.main(famFile, dataDir, Path(args.output_dir + "/dataset_gain_1day"), "first_sensor_value_gain",
                                1))


        dataset_files_li.append(
            createDataSets.main(famFile, dataDir, Path(args.output_dir + "/dataset_li_7day"), "first_sensor_value_li",
                                7))
        dataset_files_li.append(
            createDataSets.main(famFile, dataDir, Path(args.output_dir + "/dataset_li_6day"), "first_sensor_value_li",
                                6))
        dataset_files_li.append(
            createDataSets.main(famFile, dataDir, Path(args.output_dir + "/dataset_li_5day"), "first_sensor_value_li",
                                5))
        dataset_files_li.append(
            createDataSets.main(famFile, dataDir, Path(args.output_dir + "/dataset_li_4day"), "first_sensor_value_li",
                                4))
        dataset_files_li.append(
            createDataSets.main(famFile, dataDir, Path(args.output_dir + "/dataset_li_3day"), "first_sensor_value_li",
                                3))
        dataset_files_li.append(
            createDataSets.main(famFile, dataDir, Path(args.output_dir + "/dataset_li_2day"), "first_sensor_value_li",
                                2))
        dataset_files_li.append(
            createDataSets.main(famFile, dataDir, Path(args.output_dir + "/dataset_li_1day"), "first_sensor_value_li",
                                1))


    # dataset_files = ['F:\\Data2\\job_debug\\dataset_gain_7day', 'F:\\Data2\\job_debug\\dataset_gain_6day',
    #                  'F:\\Data2\\job_debug\\dataset_gain_5day', 'F:\\Data2\\job_debug\\dataset_gain_4day',
    #                  'F:\\Data2\\job_debug\\dataset_gain_3day', 'F:\\Data2\\job_debug\\dataset_gain_2day',
    #                  'F:\\Data2\\job_debug\\dataset_gain_1day']

    print(dataset_files_gain, dataset_files_li)
    print("dataset creation done.")

    dataset_exist, temperature_files, humidity_files = weather_files_exist(args.output_dir)
    if not dataset_exist:
        print("create weather data.")
        humidity_files = []
        temperature_files = []
        for d_f in dataset_files_gain:
            files = glob.glob(d_f + "/*.csv")
            t_f, h_f = add_exogenous.main(files[0], d_f, args.weather_file)
            temperature_files.append(t_f)
            humidity_files.append(h_f)
    print("weather data creation done.")

    steps = [
             # ["TEMPERATURE", "STANDARDSCALER"],
             # ["HUMIDITY", "STANDARDSCALER"],
             ["QN", "ANSCOMBE", "LOG"],
             # ["QN", "ANSCOMBE", "LOG", "HUMIDITYAPPEND", "STANDARDSCALER"],
             # ["QN", "ANSCOMBE", "LOG", "TEMPERATUREAPPEND", "STANDARDSCALER"],
             ["QN", "ANSCOMBE", "LOG", "CENTER", "STFT", "STANDARDSCALER"],
             ["QN", "ANSCOMBE", "LOG", "CENTER", "CWT(MEXH)", "STANDARDSCALER"],
             ["QN", "ANSCOMBE", "LOG", "CENTER", "CWT(MORL)", "STANDARDSCALER"],
             ]

    ml.main(steps, args.output_dir + "/ml/ml_kfold_2to2_7day", dataset_files_gain[0], "1To1", "2To2", False, 12, humidity_files[0], temperature_files[0], 5, 10, 20, 6, True, True, "RepeatedKFold", 6, 60)
    ml.main(steps, args.output_dir + "/ml/ml_kfold_2to2_6day", dataset_files_gain[1], "1To1", "2To2", False, 12, humidity_files[1], temperature_files[1], 5, 10, 20, 6, True, True, "RepeatedKFold", 6, 60)
    ml.main(steps, args.output_dir + "/ml/ml_kfold_2to2_5day", dataset_files_gain[2], "1To1", "2To2", False, 12, humidity_files[2], temperature_files[2], 5, 10, 20, 6, True, True, "RepeatedKFold", 6, 60)
    ml.main(steps, args.output_dir + "/ml/ml_kfold_2to2_4day", dataset_files_gain[3], "1To1", "2To2", False, 12, humidity_files[3], temperature_files[3], 5, 10, 20, 6, True, True, "RepeatedKFold", 6, 60)
    ml.main(steps, args.output_dir + "/ml/ml_kfold_2to2_3day", dataset_files_gain[4], "1To1", "2To2", False, 12, humidity_files[4], temperature_files[4], 5, 10, 20, 6, True, True, "RepeatedKFold", 6, 60)
    ml.main(steps, args.output_dir + "/ml/ml_kfold_2to2_2day", dataset_files_gain[5], "1To1", "2To2", False, 12, humidity_files[5], temperature_files[5], 5, 10, 20, 6, True, True, "RepeatedKFold", 6, 60)
    ml.main(steps, args.output_dir + "/ml/ml_kfold_2to2_1day", dataset_files_gain[6], "1To1", "2To2", False, 12, humidity_files[6], temperature_files[6], 5, 10, 20, 6, True, True, "RepeatedKFold", 6, 60)


    # ml.main(steps, args.output_dir + "/ml/ml_kfold_2to3_7day", dataset_files_gain[0], "1To1", "2To3", False, 12, humidity_files[0], temperature_files[0], 5, 10, 20, 6, True, True, "RepeatedKFold", 6, 60)
    # ml.main(steps, args.output_dir + "/ml/ml_kfold_2to3_6day", dataset_files_gain[1], "1To1", "2To3", False, 12, humidity_files[1], temperature_files[1], 5, 10, 20, 6, True, True, "RepeatedKFold", 6, 60)
    # ml.main(steps, args.output_dir + "/ml/ml_kfold_2to3_5day", dataset_files_gain[2], "1To1", "2To3", False, 12, humidity_files[2], temperature_files[2], 5, 10, 20, 6, True, True, "RepeatedKFold", 6, 60)
    # ml.main(steps, args.output_dir + "/ml/ml_kfold_2to3_4day", dataset_files_gain[3], "1To1", "2To3", False, 12, humidity_files[3], temperature_files[3], 5, 10, 20, 6, True, True, "RepeatedKFold", 6, 60)
    # ml.main(steps, args.output_dir + "/ml/ml_kfold_2to3_3day", dataset_files_gain[4], "1To1", "2To3", False, 12, humidity_files[4], temperature_files[4], 5, 10, 20, 6, True, True, "RepeatedKFold", 6, 60)
    # ml.main(steps, args.output_dir + "/ml/ml_kfold_2to3_2day", dataset_files_gain[5], "1To1", "2To3", False, 12, humidity_files[5], temperature_files[5], 5, 10, 20, 6, True, True, "RepeatedKFold", 6, 60)
    # ml.main(steps, args.output_dir + "/ml/ml_kfold_2to3_1day", dataset_files_gain[6], "1To1", "2To3", False, 12, humidity_files[6], temperature_files[6], 5, 10, 20, 6, True, True, "RepeatedKFold", 6, 60)

    # ml 1To1 1To2
    # ml.main(steps, args.output_dir + "/ml/ml_kfold_1to2_7day", dataset_files_gain[0], 1, 2, False, 12, humidity_files[0], temperature_files[0], 5, 10, 20, 6, True, True, "RepeatedKFold", 6, 60)
    # ml.main(steps, args.output_dir + "/ml/ml_kfold_1to2_6day", dataset_files_gain[1], 1, 2, False, 12, humidity_files[1], temperature_files[1], 5, 10, 20, 6, True, True, "RepeatedKFold", 6, 60)
    # ml.main(steps, args.output_dir + "/ml/ml_kfold_1to2_5day", dataset_files_gain[2], 1, 2, False, 12, humidity_files[2], temperature_files[2], 5, 10, 20, 6, True, True, "RepeatedKFold", 6, 60)
    # ml.main(steps, args.output_dir + "/ml/ml_kfold_1to2_4day", dataset_files_gain[3], 1, 2, False, 12, humidity_files[3], temperature_files[3], 5, 10, 20, 6, True, True, "RepeatedKFold", 6, 60)
    # ml.main(steps, args.output_dir + "/ml/ml_kfold_1to2_3day", dataset_files_gain[4], 1, 2, False, 12, humidity_files[4], temperature_files[4], 5, 10, 20, 6, True, True, "RepeatedKFold", 6, 60)
    # ml.main(steps, args.output_dir + "/ml/ml_kfold_1to2_2day", dataset_files_gain[5], 1, 2, False, 12, humidity_files[5], temperature_files[5], 5, 10, 20, 6, True, True, "RepeatedKFold", 6, 60)
    # ml.main(steps, args.output_dir + "/ml/ml_kfold_1to2_1day", dataset_files_gain[6], 1, 2, False, 12, humidity_files[6], temperature_files[6], 5, 10, 20, 6, True, True, "RepeatedKFold", 6, 60)


    # for window in [2, 30, 60, 60*12, 60*24]:
    #     ml.main(["QN", "ANSCOMBE", "LOG", "CENTER", "STFT", "STANDARDSCALER"], args.output_dir + "/ml/ml_kfold_2to2_7day_window_%d" % window, dataset_files[0], 1, 4, False, 1, None,
    #             None, 5, 10, 20, 6, True, True, "RepeatedKFold", None, window)
    #     ml.main(["QN", "ANSCOMBE", "LOG", "CENTER", "STFT", "STANDARDSCALER"], args.output_dir + "/ml/ml_kfold_2to2_6day_window_%d" % window, dataset_files[1], 1, 4, False, 1, None,
    #             None, 5, 10, 20, 6, True, True, "RepeatedKFold", None, window)
    #     ml.main(["QN", "ANSCOMBE", "LOG", "CENTER", "STFT", "STANDARDSCALER"], args.output_dir + "/ml/ml_kfold_2to2_5day_window_%d" % window, dataset_files[2], 1, 4, False, 1, None,
    #             None, 5, 10, 20, 6, True, True, "RepeatedKFold", None, window)
    #     ml.main(["QN", "ANSCOMBE", "LOG", "CENTER", "STFT", "STANDARDSCALER"], args.output_dir + "/ml/ml_kfold_2to2_4day_window_%d" % window, dataset_files[3], 1, 4, False, 1, None,
    #             None, 5, 10, 20, 6, True, True, "RepeatedKFold", None, window)
    #     ml.main(["QN", "ANSCOMBE", "LOG", "CENTER", "STFT", "STANDARDSCALER"], args.output_dir + "/ml/ml_kfold_2to2_3day_window_%d" % window, dataset_files[4], 1, 4, False, 1, None,
    #             None, 5, 10, 20, 6, True, True, "RepeatedKFold", None, window)
    #     ml.main(["QN", "ANSCOMBE", "LOG", "CENTER", "STFT", "STANDARDSCALER"], args.output_dir + "/ml/ml_kfold_2to2_2day_window_%d" % window, dataset_files[5], 1, 4, False, 1, None,
    #             None, 5, 10, 20, 6, True, True, "RepeatedKFold", None, window)
    #     ml.main(["QN", "ANSCOMBE", "LOG", "CENTER", "STFT", "STANDARDSCALER"], args.output_dir + "/ml/ml_kfold_2to2_1day_window_%d" % window, dataset_files[6], 1, 4, False, 1, None,
    #             None, 5, 10, 20, 6, True, True, "RepeatedKFold", None, window)
    #
    # for f0 in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    #     f0_str = str(f0).replace(".", "_")
    #     ml.main(["QN", "ANSCOMBE", "LOG", "CENTER", "CWT", "STANDARDSCALER"], args.output_dir + "/ml/ml_kfold_2to2_7day_wf0_%s" % f0_str, dataset_files[0], 1, 4, False, 1, None,
    #             None, 5, 10, 20, 6, True, True, "RepeatedKFold", f0, 256)
    #     ml.main(["QN", "ANSCOMBE", "LOG", "CENTER", "CWT", "STANDARDSCALER"], args.output_dir + "/ml/ml_kfold_2to2_6day_wf0_%s" % f0_str, dataset_files[1], 1, 4, False, 1, None,
    #             None, 5, 10, 20, 6, True, True, "RepeatedKFold", f0, 256)
    #     ml.main(["QN", "ANSCOMBE", "LOG", "CENTER", "CWT", "STANDARDSCALER"], args.output_dir + "/ml/ml_kfold_2to2_5day_wf0_%s" % f0_str, dataset_files[2], 1, 4, False, 1, None,
    #             None, 5, 10, 20, 6, True, True, "RepeatedKFold", f0, 256)
    #     ml.main(["QN", "ANSCOMBE", "LOG", "CENTER", "CWT", "STANDARDSCALER"], args.output_dir + "/ml/ml_kfold_2to2_4day_wf0_%s" % f0_str, dataset_files[3], 1, 4, False, 1, None,
    #             None, 5, 10, 20, 6, True, True, "RepeatedKFold", f0, 256)
    #     ml.main(["QN", "ANSCOMBE", "LOG", "CENTER", "CWT", "STANDARDSCALER"], args.output_dir + "/ml/ml_kfold_2to2_3day_wf0_%s" % f0_str, dataset_files[4], 1, 4, False, 1, None,
    #             None, 5, 10, 20, 6, True, True, "RepeatedKFold", f0, 256)
    #     ml.main(["QN", "ANSCOMBE", "LOG", "CENTER", "CWT", "STANDARDSCALER"], args.output_dir + "/ml/ml_kfold_2to2_2day_wf0_%s" % f0_str, dataset_files[5], 1, 4, False, 1, None,
    #             None, 5, 10, 20, 6, True, True, "RepeatedKFold", f0, 256)
    #     ml.main(["QN", "ANSCOMBE", "LOG", "CENTER", "CWT", "STANDARDSCALER"], args.output_dir + "/ml/ml_kfold_2to2_1day_wf0_%s" % f0_str, dataset_files[6], 1, 4, False, 1, None,
    #             None, 5, 10, 20, 6, True, True, "RepeatedKFold", f0, 256)

    # ml.main(args.output_dir + "/ml/ml_l2out_2to2_7day", dataset_files[0], 1, 4, False, 1, None, None, 5, 10, 20, 6, True, True, "LeaveTwoOut")
    # ml.main(args.output_dir + "/ml/ml_l2out_2to2_6day", dataset_files[1], 1, 4, False, 1, None, None, 5, 10, 20, 6, True, True, "LeaveTwoOut")
    # ml.main(args.output_dir + "/ml/ml_l2out_2to2_5day", dataset_files[2], 1, 4, False, 1, None, None, 5, 10, 20, 6, True, True, "LeaveTwoOut")
    # ml.main(args.output_dir + "/ml/ml_l2out_2to2_4day", dataset_files[3], 1, 4, False, 1, None, None, 5, 10, 20, 6, True, True, "LeaveTwoOut")
    # ml.main(args.output_dir + "/ml/ml_l2out_2to2_3day", dataset_files[4], 1, 4, False, 1, None, None, 5, 10, 20, 6, True, True, "LeaveTwoOut")
    # ml.main(args.output_dir + "/ml/ml_l2out_2to2_2day", dataset_files[5], 1, 4, False, 1, None, None, 5, 10, 20, 6, True, True, "LeaveTwoOut")
    # ml.main(args.output_dir + "/ml/ml_l2out_2to2_1day", dataset_files[6], 1, 4, False, 1, None, None, 5, 10, 20, 6, True, True, "LeaveTwoOut")
    #
    # ml.main(args.output_dir + "/ml/ml_l1out_2to2_7day", dataset_files[0], 1, 4, False, 1, None, None, 5, 10, 20, 6, True, True, "LeaveOneOut")
    # ml.main(args.output_dir + "/ml/ml_l1out_2to2_6day", dataset_files[1], 1, 4, False, 1, None, None, 5, 10, 20, 6, True, True, "LeaveOneOut")
    # ml.main(args.output_dir + "/ml/ml_l1out_2to2_5day", dataset_files[2], 1, 4, False, 1, None, None, 5, 10, 20, 6, True, True, "LeaveOneOut")
    # ml.main(args.output_dir + "/ml/ml_l1out_2to2_4day", dataset_files[3], 1, 4, False, 1, None, None, 5, 10, 20, 6, True, True, "LeaveOneOut")
    # ml.main(args.output_dir + "/ml/ml_l1out_2to2_3day", dataset_files[4], 1, 4, False, 1, None, None, 5, 10, 20, 6, True, True, "LeaveOneOut")
    # ml.main(args.output_dir + "/ml/ml_l1out_2to2_2day", dataset_files[5], 1, 4, False, 1, None, None, 5, 10, 20, 6, True, True, "LeaveOneOut")
    # ml.main(args.output_dir + "/ml/ml_l1out_2to2_1day", dataset_files[6], 1, 4, False, 1, None, None, 5, 10, 20, 6, True, True, "LeaveOneOut")

    # ml 1To1 1To2
    # ml.main(args.output_dir + "/ml/ml_kfold_1to2_7day", dataset_files[0], 1, 2, False, 1, None, None, 5, 10, 20, 6, True, True, "RepeatedKFold")
    # ml.main(args.output_dir + "/ml/ml_kfold_1to2_6day", dataset_files[1], 1, 2, False, 1, None, None, 5, 10, 20, 6, True, True, "RepeatedKFold")
    # ml.main(args.output_dir + "/ml/ml_kfold_1to2_5day", dataset_files[2], 1, 2, False, 1, None, None, 5, 10, 20, 6, True, True, "RepeatedKFold")
    # ml.main(args.output_dir + "/ml/ml_kfold_1to2_4day", dataset_files[3], 1, 2, False, 1, None, None, 5, 10, 20, 6, True, True, "RepeatedKFold")
    # ml.main(args.output_dir + "/ml/ml_kfold_1to2_3day", dataset_files[4], 1, 2, False, 1, None, None, 5, 10, 20, 6, True, True, "RepeatedKFold")
    # ml.main(args.output_dir + "/ml/ml_kfold_1to2_2day", dataset_files[5], 1, 2, False, 1, None, None, 5, 10, 20, 6, True, True, "RepeatedKFold")
    # ml.main(args.output_dir + "/ml/ml_kfold_1to2_1day", dataset_files[6], 1, 2, False, 1, None, None, 5, 10, 20, 6, True, True, "RepeatedKFold")

    # ml.main(args.output_dir + "/ml/ml_l2out_1to2_7day", dataset_files[0], 1, 2, False, 1, None, None, 5, 10, 20, 6, True, True, "LeaveTwoOut")
    # ml.main(args.output_dir + "/ml/ml_l2out_1to2_6day", dataset_files[1], 1, 2, False, 1, None, None, 5, 10, 20, 6, True, True, "LeaveTwoOut")
    # ml.main(args.output_dir + "/ml/ml_l2out_1to2_5day", dataset_files[2], 1, 2, False, 1, None, None, 5, 10, 20, 6, True, True, "LeaveTwoOut")
    # ml.main(args.output_dir + "/ml/ml_l2out_1to2_4day", dataset_files[3], 1, 2, False, 1, None, None, 5, 10, 20, 6, True, True, "LeaveTwoOut")
    # ml.main(args.output_dir + "/ml/ml_l2out_1to2_3day", dataset_files[4], 1, 2, False, 1, None, None, 5, 10, 20, 6, True, True, "LeaveTwoOut")
    # ml.main(args.output_dir + "/ml/ml_l2out_1to2_2day", dataset_files[5], 1, 2, False, 1, None, None, 5, 10, 20, 6, True, True, "LeaveTwoOut")
    # ml.main(args.output_dir + "/ml/ml_l2out_1to2_1day", dataset_files[6], 1, 2, False, 1, None, None, 5, 10, 20, 6, True, True, "LeaveTwoOut")
    #
    # ml.main(args.output_dir + "/ml/ml_l1out_1to2_7day", dataset_files[0], 1, 2, False, 1, None, None, 5, 10, 20, 6, True, True, "LeaveOneOut")
    # ml.main(args.output_dir + "/ml/ml_l1out_1to2_6day", dataset_files[1], 1, 2, False, 1, None, None, 5, 10, 20, 6, True, True, "LeaveOneOut")
    # ml.main(args.output_dir + "/ml/ml_l1out_1to2_5day", dataset_files[2], 1, 2, False, 1, None, None, 5, 10, 20, 6, True, True, "LeaveOneOut")
    # ml.main(args.output_dir + "/ml/ml_l1out_1to2_4day", dataset_files[3], 1, 2, False, 1, None, None, 5, 10, 20, 6, True, True, "LeaveOneOut")
    # ml.main(args.output_dir + "/ml/ml_l1out_1to2_3day", dataset_files[4], 1, 2, False, 1, None, None, 5, 10, 20, 6, True, True, "LeaveOneOut")
    # ml.main(args.output_dir + "/ml/ml_l1out_1to2_2day", dataset_files[5], 1, 2, False, 1, None, None, 5, 10, 20, 6, True, True, "LeaveOneOut")
    # ml.main(args.output_dir + "/ml/ml_l1out_1to2_1day", dataset_files[6], 1, 2, False, 1, None, None, 5, 10, 20, 6, True, True, "LeaveOneOut")

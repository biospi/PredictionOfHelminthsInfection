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
import os
from pathlib import Path

import imputation
from commands import createDataSets
import ml
import numpy as np

def imputed_data_exists(path):
    folders = [x[0] for x in os.walk(path) if "miss_rate" in x[0]]
    return len(folders) > 0, folders[0]


def famacha_dataset_exists(path):
    folders = [x[0] for x in os.walk(path) if "dataset" in x[0]]
    return len(folders) > 0, folders[::-1]


if __name__ == "__main__":
    print("starting job...")
    print("imputation...")

    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('fam_file', help='Famacha HDF5 file', type=str)

    parser.add_argument('--batch_size', help='for imputation, the number of samples in mini-batch', default=128,
                        type=int)
    parser.add_argument('--hint_rate', help='for imputation, hint probability', default=0.9, type=float)
    parser.add_argument('--alpha', help='for imputation, hyperparameter', default=100, type=float)
    parser.add_argument('--iterations', help='for imputation, number of training interations', default=600, type=int)
    parser.add_argument('--miss_rate', help='for imputation, missing data probability', default=0.0, type=float)
    parser.add_argument('--n_job', type=int, default=6, help='Number of thread to use.')
    parser.add_argument('--n_top_traces', type=int, default=17,
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
                        default=100, type=int)
    parser.add_argument('--thresh_nan_ratio', help="for imputation, max percent of nan allowed in 1 day.", default=80,
                        type=int)

    parser.add_argument('--data_col',
                        help="Name of data column in imputed file (first_sensor_value_gain | first_sensor_value | first_sensor_value_li)",
                        type=str, default='first_sensor_value_gain')
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

    dataset_exist, dataset_paths = famacha_dataset_exists(args.output_dir)

    if not dataset_exist:
        famFile = Path(args.fam_file)
        dataDir = Path(imputed_data_dir)
        dataset_files = []
        dataset_files.append(
            createDataSets.main(famFile, dataDir, Path(args.output_dir + "/dataset_gain_7day"), "first_sensor_value_gain",
                                7))
        dataset_files.append(
            createDataSets.main(famFile, dataDir, Path(args.output_dir + "/dataset_gain_6day"), "first_sensor_value_gain",
                                6))
        dataset_files.append(
            createDataSets.main(famFile, dataDir, Path(args.output_dir + "/dataset_gain_5day"), "first_sensor_value_gain",
                                5))
        dataset_files.append(
            createDataSets.main(famFile, dataDir, Path(args.output_dir + "/dataset_gain_4day"), "first_sensor_value_gain",
                                4))
        dataset_files.append(
            createDataSets.main(famFile, dataDir, Path(args.output_dir + "/dataset_gain_3day"), "first_sensor_value_gain",
                                3))
        dataset_files.append(
            createDataSets.main(famFile, dataDir, Path(args.output_dir + "/dataset_gain_2day"), "first_sensor_value_gain",
                                2))
        dataset_files.append(
            createDataSets.main(famFile, dataDir, Path(args.output_dir + "/dataset_gain_1day"), "first_sensor_value_gain",
                                1))
    else:
        print("dataset folder exists! Skipping to ml...")
        dataset_files = dataset_paths


    # dataset_files = ['F:\\Data2\\job_debug\\dataset_gain_7day', 'F:\\Data2\\job_debug\\dataset_gain_6day',
    #                  'F:\\Data2\\job_debug\\dataset_gain_5day', 'F:\\Data2\\job_debug\\dataset_gain_4day',
    #                  'F:\\Data2\\job_debug\\dataset_gain_3day', 'F:\\Data2\\job_debug\\dataset_gain_2day',
    #                  'F:\\Data2\\job_debug\\dataset_gain_1day']

    print(dataset_files)
    print("dataset creation done.")

    # ml 1To1 2To2
    ml.main(args.output_dir + "/ml/ml_kfold_2to2_7day", dataset_files[0], 1, 4, False, 1, None, None, 5, 10, 20, 6, True, True, "RepeatedStratifiedKFold", 6, 60)
    ml.main(args.output_dir + "/ml/ml_kfold_2to2_6day", dataset_files[1], 1, 4, False, 1, None, None, 5, 10, 20, 6, True, True, "RepeatedStratifiedKFold", 6, 60)
    ml.main(args.output_dir + "/ml/ml_kfold_2to2_5day", dataset_files[2], 1, 4, False, 1, None, None, 5, 10, 20, 6, True, True, "RepeatedStratifiedKFold", 6, 60)
    ml.main(args.output_dir + "/ml/ml_kfold_2to2_4day", dataset_files[3], 1, 4, False, 1, None, None, 5, 10, 20, 6, True, True, "RepeatedStratifiedKFold", 6, 60)
    ml.main(args.output_dir + "/ml/ml_kfold_2to2_3day", dataset_files[4], 1, 4, False, 1, None, None, 5, 10, 20, 6, True, True, "RepeatedStratifiedKFold", 6, 60)
    ml.main(args.output_dir + "/ml/ml_kfold_2to2_2day", dataset_files[5], 1, 4, False, 1, None, None, 5, 10, 20, 6, True, True, "RepeatedStratifiedKFold", 6, 60)
    ml.main(args.output_dir + "/ml/ml_kfold_2to2_1day", dataset_files[6], 1, 4, False, 1, None, None, 5, 10, 20, 6, True, True, "RepeatedStratifiedKFold", 6, 60)

    # ml 1To1 1To2
    ml.main(args.output_dir + "/ml/ml_kfold_1to2_7day", dataset_files[0], 1, 2, False, 1, None, None, 5, 10, 20, 6, True, True, "RepeatedStratifiedKFold", 6, 60)
    ml.main(args.output_dir + "/ml/ml_kfold_1to2_6day", dataset_files[1], 1, 2, False, 1, None, None, 5, 10, 20, 6, True, True, "RepeatedStratifiedKFold", 6, 60)
    ml.main(args.output_dir + "/ml/ml_kfold_1to2_5day", dataset_files[2], 1, 2, False, 1, None, None, 5, 10, 20, 6, True, True, "RepeatedStratifiedKFold", 6, 60)
    ml.main(args.output_dir + "/ml/ml_kfold_1to2_4day", dataset_files[3], 1, 2, False, 1, None, None, 5, 10, 20, 6, True, True, "RepeatedStratifiedKFold", 6, 60)
    ml.main(args.output_dir + "/ml/ml_kfold_1to2_3day", dataset_files[4], 1, 2, False, 1, None, None, 5, 10, 20, 6, True, True, "RepeatedStratifiedKFold", 6, 60)
    ml.main(args.output_dir + "/ml/ml_kfold_1to2_2day", dataset_files[5], 1, 2, False, 1, None, None, 5, 10, 20, 6, True, True, "RepeatedStratifiedKFold", 6, 60)
    ml.main(args.output_dir + "/ml/ml_kfold_1to2_1day", dataset_files[6], 1, 2, False, 1, None, None, 5, 10, 20, 6, True, True, "RepeatedStratifiedKFold", 6, 60)

    # for f0 in [256, 500, 600, 1440]:
    #     f0_str = str(f0).replace(".", "_")
    #     ml.main(args.output_dir + "/ml/ml_kfold_2to2_7day_wf0_%s" % f0_str, dataset_files[0], 1, 4, False, 1, None,
    #             None, 5, 10, 20, 6, True, True, "RepeatedStratifiedKFold", f0)
    #     ml.main(args.output_dir + "/ml/ml_kfold_2to2_6day_wf0_%s" % f0_str, dataset_files[1], 1, 4, False, 1, None,
    #             None, 5, 10, 20, 6, True, True, "RepeatedStratifiedKFold", f0)
    #     ml.main(args.output_dir + "/ml/ml_kfold_2to2_5day_wf0_%s" % f0_str, dataset_files[2], 1, 4, False, 1, None,
    #             None, 5, 10, 20, 6, True, True, "RepeatedStratifiedKFold", f0)
    #     ml.main(args.output_dir + "/ml/ml_kfold_2to2_4day_wf0_%s" % f0_str, dataset_files[3], 1, 4, False, 1, None,
    #             None, 5, 10, 20, 6, True, True, "RepeatedStratifiedKFold", f0)
    #     ml.main(args.output_dir + "/ml/ml_kfold_2to2_3day_wf0_%s" % f0_str, dataset_files[4], 1, 4, False, 1, None,
    #             None, 5, 10, 20, 6, True, True, "RepeatedStratifiedKFold", f0)
    #     ml.main(args.output_dir + "/ml/ml_kfold_2to2_2day_wf0_%s" % f0_str, dataset_files[5], 1, 4, False, 1, None,
    #             None, 5, 10, 20, 6, True, True, "RepeatedStratifiedKFold", f0)
    #     ml.main(args.output_dir + "/ml/ml_kfold_2to2_1day_wf0_%s" % f0_str, dataset_files[6], 1, 4, False, 1, None,
    #             None, 5, 10, 20, 6, True, True, "RepeatedStratifiedKFold", f0)

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
    # ml.main(args.output_dir + "/ml/ml_kfold_1to2_7day", dataset_files[0], 1, 2, False, 1, None, None, 5, 10, 20, 6, True, True, "RepeatedStratifiedKFold")
    # ml.main(args.output_dir + "/ml/ml_kfold_1to2_6day", dataset_files[1], 1, 2, False, 1, None, None, 5, 10, 20, 6, True, True, "RepeatedStratifiedKFold")
    # ml.main(args.output_dir + "/ml/ml_kfold_1to2_5day", dataset_files[2], 1, 2, False, 1, None, None, 5, 10, 20, 6, True, True, "RepeatedStratifiedKFold")
    # ml.main(args.output_dir + "/ml/ml_kfold_1to2_4day", dataset_files[3], 1, 2, False, 1, None, None, 5, 10, 20, 6, True, True, "RepeatedStratifiedKFold")
    # ml.main(args.output_dir + "/ml/ml_kfold_1to2_3day", dataset_files[4], 1, 2, False, 1, None, None, 5, 10, 20, 6, True, True, "RepeatedStratifiedKFold")
    # ml.main(args.output_dir + "/ml/ml_kfold_1to2_2day", dataset_files[5], 1, 2, False, 1, None, None, 5, 10, 20, 6, True, True, "RepeatedStratifiedKFold")
    # ml.main(args.output_dir + "/ml/ml_kfold_1to2_1day", dataset_files[6], 1, 2, False, 1, None, None, 5, 10, 20, 6, True, True, "RepeatedStratifiedKFold")

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

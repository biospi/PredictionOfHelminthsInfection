"""Main function for MRNN

Reference: Jinsung Yoon, William R. Zame and Mihaela van der Schaar, 
           "Estimating Missing Data in Temporal Data Streams Using 
           Multi-Directional Recurrent Neural Networks," 
           in IEEE Transactions on Biomedical Engineering, 
           vol. 66, no. 5, pp. 1477-1490, May 2019.

Paper Link: https://ieeexplore.ieee.org/document/8485748
Contact: jsyoon0823@gmail.com
--------------------------------------------------
(1) Load the data
(2) Train MRNN model
(3) Impute missing data
(4) Evaluate the imputation performance
"""

# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import warnings

from mrnn.data_loader import data_loader
from datetime import datetime

warnings.filterwarnings("ignore")
import numpy as np
import shutil
import os
import kaleido

os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices"
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from mrnn.utils import (
    imputation_performance,
    plot_rmse_performance,
    find_dense,
    stream_info,
    initialise_time_matrix,
    merge_mrnn_steps_html_figures,
)
from pathlib import Path
from sys import exit

import pandas as pd
import time
from tqdm import tqdm
import multiprocessing.dummy as mp
from mrnn import mrnn
from multiprocessing import Pool


def main(args):
    """MRNN main function.

    Args:
      - file_name: dataset file name
      - seq_len: sequence length of time-series data
      - missing_rate: the rate of introduced missingness
      - h_dim: hidden state dimensions
      - batch_size: the number of samples in mini batch
      - iteration: the number of iteration
      - learning_rate: learning rate of model training
      - metric_name: imputation performance metric (mse, mae, rmse)

    Returns:
      - output:
        - x: original data with missing
        - ori_x: original data without missing
        - m: mask matrix
        - t: time matrix
        - imputed_x: imputed data
        - performance: imputation performance
    """
    start_i = args.start_i
    end_i = args.end_i
    args.streams = np.unique(args.streams)

    streams_test = args.streams
    streams_train = args.streams
    has_weather = args.has_weather
    missing_rate = args.missing_rate
    iteration = args.iteration

    if args.cpu:
        print("force cpu!")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    dataset_name = Path(args.file_name).stem
    ## Load data
    (
        x,
        m,
        t,
        ori_x,
        data_o,
        features,
        norm_parameters,
        weather_str,
        timestamp,
        date_str,
    ) = data_loader(
        args.file_name,
        args.seq_len,
        missing_rate,
        start_i,
        end_i,
        has_weather,
        streams_test,
        args.run_id,
        args.stride,
    )

    # (
    #     x_,
    #     m_,
    #     t_,
    #     _,
    #     _,
    #     _,
    #     _,
    #     _,
    #     _,
    #     _,
    #     _,
    # ) = data_loader(
    #     args.file_name,
    #     args.seq_len,
    #     missing_rate,
    #     start_i,
    #     end_i,
    #     has_weather,
    #     streams_test,
    #     args.run_id,
    #     remove_zeros = True,
    # )
    x_train, m_train, t_train, n, features_train = find_dense(
        iteration,
        x.copy(),
        m.copy(),
        t.copy(),
        args,
        features,
        streams_train,
        args.run_id,
        args.n,
        weather_str,
    )

    if args.batch_size > n:
        args.batch_size = n

    ## Train M-RNN
    # Remove 'tmp/mrnn_imputation' directory if exist
    if os.path.exists(
        f"tmp/mrnn_imputation_{start_i}_{end_i}_{iteration}_{args.seq_len}_{n}_{args.run_id}"
    ):
        shutil.rmtree(
            f"tmp/mrnn_imputation_{start_i}_{end_i}_{iteration}_{args.seq_len}_{n}_{args.run_id}"
        )

    # mrnn model parameters
    model_parameters = {
        "h_dim": args.h_dim,
        "batch_size": args.batch_size,
        "iteration": iteration,
        "learning_rate": args.learning_rate,
    }

    # x_train, m_train, t_train = x.copy(), m.copy(), t.copy()
    # Fit mrnn_model
    mrnn_model = mrnn.mrnn(
        n,
        x_train,
        model_parameters,
        missing_rate,
        dataset_name,
        weather_str,
        start_i,
        end_i,
        args.run_id,
        args.streams,
        x_train.shape,
        args.stride,
        args.seq_len
    )
    start_time = time.time()
    print("start fitting...")
    mrnn_model.fit(x_train, m_train, t_train, features_train)
    end_time = time.time()
    fit_time = end_time - start_time
    print(f"Fitting Duration: {fit_time}")
    # if os.path.exists(
    #     f"tmp/mrnn_imputation_{start_i}_{end_i}_{iteration}_{args.seq_len}_{n}_{args.run_id}"
    # ):
    #     shutil.rmtree(
    #         f"tmp/mrnn_imputation_{start_i}_{end_i}_{iteration}_{args.seq_len}_{n}_{args.run_id}"
    #     )
    return mrnn_model, norm_parameters

    # merge_mrnn_steps_html_figures(
    #     "training",
    #     0,
    #     args.run_id,
    #     missing_rate,
    #     args.seq_len,
    #     iteration,
    #     weather_str,
    #     n
    # )
    # print("start testing...")
    # # Impute missing data
    # start_time = time.time()
    # # testing data must be the same shape as the training data!
    # imputed_x = []
    # x_test_list = []
    # m_test_list = []
    # # t_test_list = []
    #
    # range1 = np.arange(0, x.shape[0], x_train.shape[0])
    # if range1[-1] < x.shape[0]:
    #     range1 = np.append(range1, x.shape[0])
    #
    # range2 = np.arange(0, x.shape[2], x_train.shape[2])
    # if range2[-1] < x.shape[2]:
    #     range2 = np.append(range2, x.shape[2])
    #
    # t_test_ = initialise_time_matrix(np.empty(x_train.shape))
    #
    # cpt_empty_block = 0
    # last_fully_synth = None
    # cpt = 0
    # for i in range(len(range1) - 1):
    #     imputed_streams = []
    #     x_streams = []
    #     m_streams = []
    #     for j in range(len(range2) - 1):
    #         print(
    #             f"progress {i}/{len(range1) - 1} {j}/{len(range2) - 1}..."
    #         )
    #         x_test = np.zeros(x_train.shape)
    #         m_test = np.zeros(x_train.shape)
    #         t_test = t_test_.copy()
    #
    #         data_x = x[
    #             range1[i] : range1[i + 1], :, range2[j] : range2[j + 1]
    #         ]
    #         x_test[
    #             : data_x.shape[0], : data_x.shape[1], : data_x.shape[2]
    #         ] = data_x
    #
    #         data_m = m[
    #             range1[i] : range1[i + 1], :, range2[j] : range2[j + 1]
    #         ]
    #         m_test[
    #             : data_m.shape[0], : data_m.shape[1], : data_m.shape[2]
    #         ] = data_m
    #
    #         data_t = t[
    #             range1[i] : range1[i + 1], :, range2[j] : range2[j + 1]
    #         ]
    #         t_test[
    #             : data_t.shape[0], : data_t.shape[1], : data_t.shape[2]
    #         ] = data_t
    #
    #         # if np.all(x_test == 0):
    #         #     cpt_empty_block += 1
    #         #
    #         # if not np.all(x_test == 0):
    #         #     cpt_empty_block = 0
    #         #
    #         # if cpt_empty_block > 2:
    #         #     print("last 2 testing window was empty, use last synthetic prediction")
    #         #     if last_fully_synth is None:
    #         #         print("ERROR! No previous entry")
    #         #         last_fully_synth = mrnn_model.transform(x_test, m_test, t_test, i, tag="reshaped_testing")
    #         #
    #         #     im_x = last_fully_synth
    #         # else:
    #         #     im_x = mrnn_model.transform(x_test, m_test, t_test, i, tag="reshaped_testing")
    #         #     last_fully_synth = im_x
    #
    #         im_x = mrnn_model.transform(x_test.copy(), m_test.copy(), t_test.copy(), cpt, tag="reshaped_testing")
    #         cpt += 1
    #
    #         # x_test_list.append(x_test)
    #         # m_test_list.append(m_test)
    #         # t_test_list.append(t_test)
    #
    #         im_x = im_x[
    #             : data_x.shape[0], : data_x.shape[1], : data_x.shape[2]
    #         ]
    #         x_test = x_test[
    #             : data_x.shape[0], : data_x.shape[1], : data_x.shape[2]
    #         ]
    #         m_test = m_test[
    #             : data_x.shape[0], : data_x.shape[1], : data_x.shape[2]
    #         ]
    #
    #         # for q in range(im_x.shape[0]):
    #         #     if (m_test[q, :, :] == 0).all():
    #         #         im_x[q, :, :] = np.nan
    #
    #         p = 720
    #         # imputed_streams.append(im_x[:, p:-p, :])
    #         # x_streams.append(x_test[:, p:-p, :])
    #         # m_streams.append(m_test[:, p:-p, :])
    #         imputed_streams.append(im_x)
    #         x_streams.append(x_test)
    #         m_streams.append(m_test)
    #
    #     imputed_streams = np.dstack(imputed_streams)
    #     imputed_x.append(imputed_streams)
    #
    #     x_streams = np.dstack(x_streams)
    #     x_test_list.append(x_streams)
    #
    #     m_streams = np.dstack(m_streams)
    #     m_test_list.append(m_streams)
    #     merge_mrnn_steps_html_figures(
    #         "testing",
    #         i,
    #         args.run_id,
    #         missing_rate,
    #         args.seq_len,
    #         iteration,
    #         weather_str,
    #         n
    #     )
    #
    # imputed_x = np.vstack(np.array(imputed_x))
    # ori_x = np.vstack(np.array(x_test_list))
    # m = np.vstack(np.array(m_test_list))
    #
    # end_time = time.time()
    # impute_time = end_time - start_time
    # print(f"Imputation Duration: {impute_time}")
    #
    # # Evaluate the imputation performance
    # # print(ori_x[0])
    # print("***********")
    # # print(imputed_x[0])
    # performance, performance_li, imputation_output_dir = imputation_performance(
    #     args.streams,
    #     args.run_id,
    #     n,
    #     data_mask,
    #     ori_x,
    #     imputed_x,
    #     m,
    #     args.metric_name,
    #     features,
    #     args,
    #     dataset_name,
    #     norm_parameters,
    #     weather_str,
    #     start_i,
    #     end_i,
    #     iteration,
    #     timestamp,
    #     date_str,
    #     args.export_csv,
    # )
    #
    # # Report the result
    # # print(args.metric_name + ": " + str(np.round(performance, 4)))
    #
    # # Return the output
    # output = {
    #     "run_id": args.run_id,
    #     "streams": args.streams,
    #     "n_streams": len(args.streams),
    #     "n_samples": n,
    #     "has_weather": weather_str,
    #     "dataset_name": dataset_name,
    #     "seq_len": args.seq_len,
    #     "start_i": start_i,
    #     "end_i": end_i,
    #     "missing_rate": missing_rate,
    #     "iteration": iteration,
    #     "learning_rate": args.learning_rate,
    #     "performance": performance,
    #     "performance_li": performance_li,
    #     "fit_time": fit_time,
    #     "impute_time": impute_time,
    # }
    # df = pd.DataFrame(output.items()).T
    # df = df.rename(columns=df.iloc[0]).drop(df.index[0])
    #
    # filename = f"{args.run_id}_performance_report_{dataset_name}_{start_i}_{end_i}_seql_{args.seq_len}_n_{n}_mr_{missing_rate}_iter_{iteration}_lr_{args.learning_rate}_hw_{weather_str}.csv"
    # output_dir = Path("performance")
    # output_dir.mkdir(parents=True, exist_ok=True)
    # filepath = output_dir / filename
    # df.to_csv(filepath, index=False)
    #
    # if os.path.exists(
    #     f"tmp/mrnn_imputation_{start_i}_{end_i}_{iteration}_{args.seq_len}_{n}_{args.run_id}"
    # ):
    #     shutil.rmtree(
    #         f"tmp/mrnn_imputation_{start_i}_{end_i}_{iteration}_{args.seq_len}_{n}_{args.run_id}"
    #     )
    # return imputation_output_dir


def start_mrnn(farmname, filepath, run_id=0, seq_len=1440, stride=1440, filter_training=False, iteration=100):

    print("fitting fold data with mrnn...")
    path = Path(os.path.dirname(os.path.dirname(__file__))) / "mrnn"
    print(f"set current dir to {path}")
    os.chdir(path)
    streams = []
    if farmname == "delmas":
        streams = [
            "40101310013",
            "40101310040",
            "40101310069"
            # "40101310085",
            # "40101310098"
            # "40101310109",
            # "40101310110",
            # "40101310134",
            # "40101310143",
            # "40101310249",
            # "40101310314",
            # "40101310316",
            # "40101310342",
            # "40101310350",
            # "40101310353",
            # "40101310386",
            # "40101310409"
            # "40101310040",
            # "40101310110",
            # "40101310316",
            # "40101310109",
            # "40101310085",
            # "40101310353",
            # "40101310314",
            # "40101310409",
            # "40101310143",
            # "40101310134",
            # "40101310342",
            # "40101310069",
            # "40101310013",
            # "40101310098",
            # "40101310350",
            # "40101310137",
            # "40101310121",
            # "40101310241",
            # "40101310386",
            # "40101310081",
            # "40101310249",
            # "40101310119",
            # "40101310299",
            # "40101310231",
            # "40101310094",
            # "40101310345",
            # "40101310220",
            # "40101310224",
            # "40101310247",
            # "40101310395",
            # "40101310026",
            # "40101310318",
            # "40101310389",
            # "40101310125",
            # "40101310039",
            # "40101310145",
            # "40101310238",
            # "40101310050",
            # "40101310228",
            # "40101310230",
            # "40101310142",
            # "40101310106",
            # "40101310092",
            # "40101310347",
            # "40101310071",
            # "40101310310",
            # "40101310117",
            # "40101310083",
            # "40101310036",
            # "40101310336",
            # "40101310095",
            # "40101310157",
            # "40101310332",
            # "40101310115",
            # "40101310016",
            # "40101310239",
            # "40101310229",
            # "40101310086",
            # "40101310100",
            # "40101310352",
            # "40101310135"
        ]
    if farmname == "cedara":
        streams = [
            '40011301507', '40011301509', '40011301510', '40011301511',
            '40011301512', '40011301515', '40011301517', '40011301520',
            '40011301527', '40011301528', '40011301529', '40011301539',
            '40011301542', '40011301545', '40011301546', '40011301551',
            '40011301556', '40011301557', '40011301559', '40011301565',
            '40011301568', '40011301575', '40011301581', '40011301589',
            '40011301590', '40011301591', '40011301596', '40011301597',
            '40011301599', '40061200838', '40061200862', '40061200873',
            '40061200880', '40061200904', '40061200910', '40061200922',
            '40061200928', '40061200930', '40061200934', '40061200951',
            '40061200966', '40061200978', '40061201006', '40061201015',
            '40061201018', '40061201024', '40061201049', '40061201055',
            '40061201068', '40061201073', '40061201077', '40061201098',
            '40061201134'
        ]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_name", default=filepath, type=str
    )
    parser.add_argument(
        "--streams",
        help="Stream to use for training",
        nargs="+",
        default=streams,
        type=str,
    )
    parser.add_argument(
        "--run_id",
        help="Run id",
        default=run_id,
        type=int,
    )
    parser.add_argument(
        "--n",
        help="Number of windows for training",
        default=800,
        type=int,
    )
    parser.add_argument(
        "--start_i",
        help="start index of time-series data",
        default=30240,
        type=int,
    )
    parser.add_argument(
        "--end_i",
        help="start index of time-series data",
        default=-1,
        type=int,
    )
    parser.add_argument(
        "--seq_len",
        help="sequence length of time-series data",
        default=seq_len,
        type=int,
    )
    parser.add_argument("--cpu", help="force cpu", action="store_true")
    parser.add_argument(
        "--missing_rate",
        help="the rate of introduced missingness",
        default=0.0,
        type=float,
    )
    parser.add_argument("--h_dim", help="hidden state dimensions", default=10, type=int)
    parser.add_argument(
        "--batch_size",
        help="the number of samples in mini batch",
        default=100,
        type=int,
    )

    parser.add_argument(
        "--stride",
        help="stride",
        default=stride,
        type=int,
    )

    parser.add_argument(
        "--iteration",
        help="the number of iteration",
        # default=np.arange(1, 100, 1).tolist(),
        default=iteration,
        type=int,
    )
    parser.add_argument(
        "--learning_rate",
        help="learning rate of model training",
        default=0.01,
        type=float,
    )
    parser.add_argument(
        "--metric_name",
        help="imputation performance metric",
        default="rmse",
        type=str,
    )
    parser.add_argument(
        "--export_csv",
        help="Export csv",
        default=True,
        type=bool,
    )

    parser.add_argument(
        "--filter_training",
        help="Filter training samples",
        default=filter_training,
        type=bool,
    )

    parser.add_argument(
        "--has_weather",
        help="enable weather",
        default=False,
        type=bool,
    )

    args = parser.parse_args()
    #args, unknown = parser.parse_known_args()

    model, norm_parameters = main(args)
    # print("mrrn imputation done!")
    # path = Path(os.path.dirname(os.path.dirname(__file__)))
    # print(f"set current dir to {path}")
    os.chdir(path)
    return model, norm_parameters, streams


if __name__ == "__main__":
    # plot_rmse_performance(0, -1, 464, 99)
    # # #stream_info("data/activity_data.csv")
    exit()

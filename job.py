import argparse
from pathlib import Path

import imputation
from commands import createDataSets

if __name__ == "__main__":
    print("starting job...")
    print("imputation...")

    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('fam_file', help='Famacha HDF5 file', type=str)

    parser.add_argument('--batch_size', help='for imputation, the number of samples in mini-batch', default=128, type=int)
    parser.add_argument('--hint_rate', help='for imputation, hint probability', default=0.9, type=float)
    parser.add_argument('--alpha', help='for imputation, hyperparameter', default=100, type=float)
    parser.add_argument('--iterations', help='for imputation, number of training interations', default=600, type=int)
    parser.add_argument('--miss_rate', help='for imputation, missing data probability', default=0.0, type=float)
    parser.add_argument('--n_job', type=int, default=6, help='Number of thread to use.')
    parser.add_argument('--n_top_traces', type=int, default=17,
                        help='for imputation, select n traces with highest entropy (<= 0 to select all traces)')
    parser.add_argument('--enable_anscombe', help="for imputation, appy anscombe on activity count before imputation", type=bool, default=False)
    parser.add_argument('--enable_remove_zeros', help="for imputation, remove zero counts in activity before imputation",type=bool, default=False)
    parser.add_argument('--enable_log_anscombe', help="for imputation, appy log(anscombe) on activity count before imputation",type=bool, default=True)
    parser.add_argument('--window', type=bool, default=False)
    parser.add_argument('--export_csv', type=bool, help="for imputation, export imputed traces as csv", default=True)
    parser.add_argument('--export_traces', type=bool, default=True)
    parser.add_argument('--reshape', type=str, help="for imputation, reshape activity traces to 1 day chunck", default='y')
    parser.add_argument('--w', type=str, default='n')
    parser.add_argument('--add_t_col', help="for imputation, add time column in reshape", type=str, default='y')
    parser.add_argument('--thresh_daytime', help="for imputation, minimum number of positive count in 1 day.", default=100, type=int)
    parser.add_argument('--thresh_nan_ratio', help="for imputation, max percent of nan allowed in 1 day.", default=80, type=int)


    parser.add_argument('--data_col', help="Name of data column in imputed file (first_sensor_value_gain | first_sensor_value | first_sensor_value_li)", type=str, default='first_sensor_value_gain')
    parser.add_argument('--ndays', help="Number of days in samples", default=7, type=int)

    args = parser.parse_args()

    imputed_data_dir = imputation.start(args)
    #imputed_data_dir = "F:\Data2\job_debug\miss_rate_0_0_iteration_0600_thresh_100_anscombe_False_n_top_traces_17"
    print("imputation done.")

    famFile = Path(args.fam_file)
    dataDir = Path(imputed_data_dir)

    dataset_files = []
    dataset_files.append(createDataSets.main(famFile, dataDir, Path(args.output_dir+"/dataset_gain_7day"), "first_sensor_value_gain", 7))
    dataset_files.append(createDataSets.main(famFile, dataDir, Path(args.output_dir+"/dataset_gain_6day"), "first_sensor_value_gain", 6))
    dataset_files.append(createDataSets.main(famFile, dataDir, Path(args.output_dir+"/dataset_gain_5day"), "first_sensor_value_gain", 5))
    dataset_files.append(createDataSets.main(famFile, dataDir, Path(args.output_dir+"/dataset_gain_4day"), "first_sensor_value_gain", 4))
    dataset_files.append(createDataSets.main(famFile, dataDir, Path(args.output_dir+"/dataset_gain_3day"), "first_sensor_value_gain", 3))
    dataset_files.append(createDataSets.main(famFile, dataDir, Path(args.output_dir+"/dataset_gain_2day"), "first_sensor_value_gain", 2))
    dataset_files.append(createDataSets.main(famFile, dataDir, Path(args.output_dir+"/dataset_gain_1day"), "first_sensor_value_gain", 1))
    print(dataset_files)

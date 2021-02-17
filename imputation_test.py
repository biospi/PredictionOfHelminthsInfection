import argparse
import imputation as imputation
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(0) #for reproducability
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--reshape', type=str)
    parser.add_argument('--w', type=str)
    args = parser.parse_args()
    print(args)

    NTOP = 17
    NJOB = 6
    ANSCOMBE = False
    LOG_ANSCOMBE = True
    REMOVE_ZEROS = True
    EXPORT_CSV = False
    EXPORT_TRACES = False
    ENABLE_FINAL_IMP = False
    if ENABLE_FINAL_IMP:
        EXPORT_CSV = True
    WINDOW_ON = args.w.lower() in ["yes", 'y', 't', 'true']
    RESHAPE = args.reshape.lower() in ["yes", 'y', 't', 'true']
    OUT = args.output_dir
    I_RANGE = 610

    # DATA_DIR = 'F:/Data2/backfill_1min_xyz_delmas_fixed'
    # DATA_DIR = 'backfill_1min_xyz_delmas_fixed'
    DATA_DIR = args.data_dir

    # config = [(WINDOW_ON, True, False, False), (WINDOW_ON, False, False, False), (WINDOW_ON, True, True, False), (WINDOW_ON, False, True, False), (WINDOW_ON, True, False, True), (WINDOW_ON, False, False, True)]
    config = [(WINDOW_ON, False, ANSCOMBE, LOG_ANSCOMBE)]
    for WINDOW_ON, REMOVE_ZEROS, ANSCOMBE, LOG_ANSCOMBE in config:

        OUT += '\imputation_test_window_%s_anscombe_%s_top%d_remove_zeros_%s_loganscombe_%s_reshape_%s' % (WINDOW_ON, ANSCOMBE, NTOP, REMOVE_ZEROS, LOG_ANSCOMBE, str(RESHAPE))
        # OUT = 'F:/Data2/imp_reshaped_full/imputation_test_window_%s_anscombe_%s_top%d_remove_zeros_%s_loganscombe_%s_debug' % (WINDOW_ON, ANSCOMBE, NTOP, REMOVE_ZEROS, LOG_ANSCOMBE)

        raw_data, original_data_x, ids, timestamp, date_str = imputation.load_farm_data(DATA_DIR, NJOB, NTOP, enable_remove_zeros=REMOVE_ZEROS,
                                                                                        enable_anscombe=ANSCOMBE, enable_log_anscombe=LOG_ANSCOMBE, window=WINDOW_ON)
        iteration_range = np.array(list(range(10, I_RANGE, 10)))
        missing_range = [0.1]
        if ENABLE_FINAL_IMP:
            missing_range = [0]
            iteration_range = [280]
            # iteration_range = [600]

        for miss_rate in missing_range:
            rmse_list = []
            rmse_list_li = []
            for i_r in iteration_range:
                parser = argparse.ArgumentParser()
                parser.add_argument('--data_dir', type=str, default=DATA_DIR)
                parser.add_argument('--output_dir', type=str, default=OUT)
                parser.add_argument(
                    '--batch_size',
                    help='the number of samples in mini-batch',
                    default=16,
                    type=int)
                parser.add_argument(
                    '--hint_rate',
                    help='hint probability',
                    default=0.9,
                    type=float)
                parser.add_argument(
                    '--alpha',
                    help='hyperparameter',
                    default=100,
                    type=float)
                parser.add_argument(
                    '--iterations',
                    help='number of training interations',
                    default=i_r,
                    type=int)
                parser.add_argument(
                    '--miss_rate',
                    help='missing data probability',
                    default=miss_rate,
                    type=float)
                parser.add_argument('--n_job', type=int, default=NJOB, help='Number of thread to use.')
                parser.add_argument('--n_top_traces', type=int, default=NTOP,
                                    help='select n traces with highest entropy (<= 0 number to select all traces)')
                parser.add_argument('--enable_anscombe', type=bool, default=ANSCOMBE)
                parser.add_argument('--export_csv', type=bool, default=EXPORT_CSV)
                parser.add_argument('--export_traces', type=bool, default=EXPORT_TRACES)
                parser.add_argument('--reshape', type=str, default=RESHAPE)
                parser.add_argument('--w', type=str, default=WINDOW_ON)

                args = parser.parse_args()
                print(args)
                imputed_data_x, rmse, rmse_li = imputation.main(args, raw_data, original_data_x, ids, timestamp, date_str)
                print(imputed_data_x, rmse, rmse_li)
                rmse_list.append(rmse)
                rmse_list_li.append(rmse_li)
            plt.clf()
            plt.cla()
            fig, ax = plt.subplots()
            ax.set_ylabel('RMSE')
            ax.set_xlabel('iteration')
            plt.plot(iteration_range, rmse_list, label="RMSE GAIN", alpha=1, marker='*')
            plt.plot(iteration_range, rmse_list_li, label="RMSE LI", alpha=1, marker='*')

            plt.title("RMSE iteration performance\nmissing rate=%d(%%) Log Anscombe=%s\n best n traces=%d" % (miss_rate*100, LOG_ANSCOMBE, NTOP))
            plt.legend()
            filename = args.output_dir + "/" + "RMSE_%d_%s.png" % (miss_rate*100, ANSCOMBE)
            print(filename)
            plt.savefig(filename)


            plt.clf()
            plt.cla()
            fig, ax = plt.subplots()
            ax.set_ylabel('RMSE')
            ax.set_xlabel('iteration')
            plt.plot(iteration_range, rmse_list_li, label="RMSE LI", alpha=1, marker='*')

            plt.title("RMSE iteration performance\nmissing rate=%d(%%) Log Anscombe=%s\n best n traces=%d" % (miss_rate*100, ANSCOMBE, NTOP))
            plt.legend()
            filename = args.output_dir + "/" + "RMSE_LI_%d_%s.png" % (miss_rate*100, ANSCOMBE)
            print(filename)
            plt.savefig(filename)

        continue


        rmse_list = []
        rmse_list_li = []
        # rmse_list_pt = []
        # rmse_list_li_pt = []
        missing_range = np.arange(0.1, 0.9, 0.1)
        # missing_range = [0.5]
        n_iteration = 1000
        for m_r in missing_range:
            parser = argparse.ArgumentParser()
            parser.add_argument('--data_dir', type=str, default=DATA_DIR)
            parser.add_argument('--output_dir', type=str, default=OUT)
            parser.add_argument(
                '--batch_size',
                help='the number of samples in mini-batch',
                default=16,
                type=int)
            parser.add_argument(
                '--hint_rate',
                help='hint probability',
                default=0.9,
                type=float)
            parser.add_argument(
                '--alpha',
                help='hyperparameter',
                default=100,
                type=float)
            parser.add_argument(
                '--iterations',
                help='number of training interations',
                default=n_iteration,
                type=int)
            parser.add_argument(
                '--miss_rate',
                help='missing data probability',
                default=m_r,
                type=float)
            parser.add_argument('--n_job', type=int, default=NJOB, help='Number of thread to use.')
            parser.add_argument('--n_top_traces', type=int, default=NTOP,
                                help='select n traces with highest entropy (<= 0 number to select all traces)')
            parser.add_argument('--enable_anscombe', type=bool, default=ANSCOMBE)
            parser.add_argument('--export_csv', type=bool, default=EXPORT_CSV)
            parser.add_argument('--export_traces', type=bool, default=EXPORT_TRACES)
            parser.add_argument('--reshape', type=bool, default=RESHAPE)

            args = parser.parse_args()
            imputed_data_x, rmse, rmse_li = imputation.main(args, raw_data, original_data_x, ids, timestamp, date_str)
            print(imputed_data_x, rmse, rmse_li)
            rmse_list.append(rmse)
            rmse_list_li.append(rmse_li)
            # rmse_list_pt.append(rmse_per_id)
            # rmse_list_li_pt.append(rmse_per_id_li)

        plt.clf()
        plt.cla()
        fig, ax = plt.subplots()
        ax.set_ylabel('RMSE')
        ax.set_xlabel('missing (%)')
        plt.plot(missing_range, rmse_list, label="RMSE GAIN", alpha=1, marker='*')
        plt.plot(missing_range, rmse_list_li, label="RMSE Linear interpolation", alpha=1, marker='*')

        plt.title("RMSE missingness performance\niteration=%d Log Anscombe=%s\n best n traces=%d" % (n_iteration, ANSCOMBE, NTOP))
        plt.legend()
        filename = args.output_dir + "/" + "RMSE_missing_%d_%s.png" % (n_iteration, ANSCOMBE)
        print(filename)
        plt.savefig(filename)


        # plt.clf()
        # plt.cla()
        # fig, ax = plt.subplots(figsize=(19.20, 10.80))
        # ax.set_ylabel('RMSE')
        # ax.set_xlabel('missing (%)')
        #
        # for c, key in enumerate(rmse_list_pt[0].keys()):
        #     data = []
        #     data_li = []
        #     for i in range(len(rmse_list_pt)):
        #         v = rmse_list_pt[i][key]
        #         data.append(v)
        #         v_li = rmse_list_li_pt[i][key]
        #         data_li.append(v_li)
        #     color = np.random.rand(3,)
        #     plt.plot(missing_range, data, label="RMSE GAIN %s" % key, alpha=1, linestyle='-', color=color)
        #     plt.plot(missing_range, data_li, label="RMSE Linear interpolation %s" % key, alpha=1, linestyle='-.', color=color)
        #
        # plt.title("RMSE missingness performance per transponder\niteration=%d Log Anscombe=%s\n best n traces=%d" % (n_iteration, ANSCOMBE, NTOP))
        # plt.legend()
        # filename = args.output_dir + "/" + "RMSE_missing_pt_%d_%s.png" % (n_iteration, ANSCOMBE)
        # print(filename)
        # plt.savefig(filename)





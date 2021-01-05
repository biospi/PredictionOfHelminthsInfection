import argparse
import gainimputation.imputation as imputation
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    NTOP = 0
    NJOB = 6
    ANSCOMBE = True
    EXPORT_CSV = False
    WINDOW_ON = True
    DATA_DIR = 'F:/Data2/backfill_1min_xyz_delmas_fixed'

    for WINDOW_ON in [True, False]:

        OUT = 'F:/Data2/imputation_test_window_%s_anscombe_%s_top%d_debug' % (WINDOW_ON, ANSCOMBE, NTOP)
        ori_data_x, ids, timestamp, date_str = imputation.load_farm_data(DATA_DIR, NJOB, NTOP,
                                                              enable_anscombe=ANSCOMBE, window=WINDOW_ON)
        # rmse_list = []
        # rmse_list_li = []
        #
        # iteration_range = np.array(list(range(10, 1000, 10)))
        # miss_rate = 0.2
        # for i_r in iteration_range:
        #     parser = argparse.ArgumentParser()
        #     parser.add_argument('--data_dir', type=str, default=DATA_DIR)
        #     parser.add_argument('--output_dir', type=str, default=OUT)
        #     parser.add_argument(
        #         '--batch_size',
        #         help='the number of samples in mini-batch',
        #         default=128,
        #         type=int)
        #     parser.add_argument(
        #         '--hint_rate',
        #         help='hint probability',
        #         default=0.9,
        #         type=float)
        #     parser.add_argument(
        #         '--alpha',
        #         help='hyperparameter',
        #         default=100,
        #         type=float)
        #     parser.add_argument(
        #         '--iterations',
        #         help='number of training interations',
        #         default=i_r,
        #         type=int)
        #     parser.add_argument(
        #         '--miss_rate',
        #         help='missing data probability',
        #         default=miss_rate,
        #         type=float)
        #     parser.add_argument('--n_job', type=int, default=NJOB, help='Number of thread to use.')
        #     parser.add_argument('--n_top_traces', type=int, default=NTOP,
        #                         help='select n traces with highest entropy (<= 0 number to select all traces)')
        #     parser.add_argument('--enable_anscombe', type=bool, default=ANSCOMBE)
        #     parser.add_argument('--export_csv', type=bool, default=EXPORT_CSV)
        #
        #     args = parser.parse_args()
        #     imputed_data_x, rmse, rmse_li = imputation.main(args, ori_data_x, ids, timestamp, date_str)
        #     print(imputed_data_x, rmse, rmse_li)
        #     rmse_list.append(rmse)
        # plt.clf()
        # plt.cla()
        # fig, ax = plt.subplots()
        # ax.set_ylabel('RMSE')
        # ax.set_xlabel('iteration')
        # plt.plot(iteration_range, rmse_list, label="RMSE GAIN", alpha=1, marker='*')
        #
        # plt.title("RMSE iteration performance\nmissing rate=%d(%%) Log Anscombe=%s\n best n traces=%d" % (miss_rate*100, ANSCOMBE, NTOP))
        # plt.legend()
        # filename = args.output_dir + "/" + "RMSE_%d_%s.png" % (miss_rate*100, ANSCOMBE)
        # print(filename)
        # plt.savefig(filename)


        rmse_list = []
        rmse_list_li = []
        missing_range = np.arange(0.1, 0.9, 0.1)
        n_iteration =1000
        for m_r in missing_range:
            parser = argparse.ArgumentParser()
            parser.add_argument('--data_dir', type=str, default=DATA_DIR)
            parser.add_argument('--output_dir', type=str, default=OUT)
            parser.add_argument(
                '--batch_size',
                help='the number of samples in mini-batch',
                default=128,
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

            args = parser.parse_args()
            imputed_data_x, rmse, rmse_li = imputation.main(args, ori_data_x, ids, timestamp, date_str)
            print(imputed_data_x, rmse, rmse_li)
            rmse_list.append(rmse)
            rmse_list_li.append(rmse_li)

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


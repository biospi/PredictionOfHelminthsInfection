import glob
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pathlib
import glob2

if __name__ == "__main__":
    print("args: output_folder datset_parent_folder")
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
        dataset_folder = sys.argv[2]
    else:
        exit(-1)

    print("output_dir=", output_dir)
    print("dataset_folder=", dataset_folder)
    print("searching txt files...")
    files = glob2.glob(dataset_folder)
    print("found %d .txt files" % len(files))
    filter_files = []
    for file in files:
        file = file.replace("\\", '/')
        filename = file.split('/')[-1]
        if len(filename) > 45:
            filter_files.append(file)

    if len(filter_files) == 0:
        print("no final files in %s" % dataset_folder)
        exit(-1)
    print("found %d files" % len(filter_files))
    farm_id = ""
    df = pd.DataFrame(columns=['usable_11', 'usable_12', 'n_days_before_famacha', 'thresh_interpol', 'thresh_zero', 'resolution'])
    for i, file in enumerate(filter_files):
        print("file=", file)
        with open(file) as f:
            lines = f.readlines()
            try:
                usable_11 = int(lines[7].split('=')[1].strip())
                usable_12 = int(lines[8].split('=')[1].strip())
                split = file.split("/")[-1].split('_')
                n_days_before_famacha = int(split[8])
                thresh_interpol = int(split[10])
                thresh_zero = int(split[12].replace(".txt", ""))
                farm_id = split[4] + "_" + split[5]
                resolution = file.split('/')[-2].split("_")[2]
            except Exception as e:
                print("resolution=", resolution)
                print("split=", split)
                print("farm_id=", farm_id)
                print("usable_11=", usable_11)
                print("usable_12=", usable_12)
                print("n_days_before_famacha=", n_days_before_famacha)
                print("thresh_interpol=", thresh_interpol)
                print("thresh_zero=", thresh_zero)
                print("resolution=", resolution)
            df.loc[i] = [usable_11, usable_12, n_days_before_famacha, thresh_interpol, thresh_zero, resolution]

    print("DF")
    print(df)
    list_of_df_grouped_by_resolution = [g for _, g in df.groupby(['resolution'])]

    try:
        print("mkdir=", output_dir)
        pathlib.Path(output_dir).mkdir(parents=True)
    except Exception as e:
        print(e)
    try:
        pathlib.Path(output_dir + "/svg").mkdir(parents=True)
    except Exception as e:
        print(e)
    try:
        print("mkdir=", output_dir)
        pathlib.Path(output_dir + "/png").mkdir(parents=True)
    except Exception as e:
        print(e)

    # fig, axs = plt.subplots(1, len(list_of_df), figsize=(18., 7.2))
    # for i, data_frame in enumerate(list_of_df):
    #     data_frame = data_frame.sort_values(by=['thresh_zero'])
    #     dbt = int(data_frame["n_days_before_famacha"].values[0])
    #     resolution = data_frame["resolution"].values[0]
    #     axs[i].plot(data_frame["thresh_zero"], data_frame["usable_11"], label="usable_11")
    #     axs[i].plot(data_frame["thresh_zero"], data_frame["usable_12"], label="usable_12")
    #     axs[i].set_title("sampling=%s days before test=%d" % (resolution, dbt))
    #     axs[i].set(xlabel='Threshold zeros')
    #     axs[i].legend(loc="upper left")
    # # plt.show()
    # out_filename = "%s/%s_thresh_zero_%s_threshi_%d_thresh_z%d.png" % (resolution, output_dir, farm_id, thresh_interpol, thresh_zero)
    # fig.savefig(out_filename)
    # fig.savefig(out_filename.replace(".png", ".svg"))
    #
    # plt.close(fig)
    # plt.clf()
    # fig, axs = plt.subplots(1, len(list_of_df), figsize=(18., 7.2))
    # for i, data_frame in enumerate(list_of_df):
    #     data_frame = data_frame.sort_values(by=['thresh_interpol'])
    #     dbt = int(data_frame["n_days_before_famacha"].values[0])
    #     resolution = data_frame["resolution"].values[0]
    #     axs[i].plot(data_frame["thresh_interpol"], data_frame["usable_11"], label="usable_11")
    #     axs[i].plot(data_frame["thresh_interpol"], data_frame["usable_12"], label="usable_12")
    #     axs[i].set_title("sampling=%s days before test=%d" % (resolution, dbt))
    #     axs[i].set(xlabel='Threshold interpolation')
    #     axs[i].legend(loc="upper left")
    # # plt.show()
    # out_filename = "%s/%s_thresh_interpol_%s_threshi_%d_thresh_z%d.png" % (
    # output_dir, resolution, farm_id, thresh_interpol, thresh_zero)
    # fig.savefig(out_filename)
    # fig.savefig(out_filename.replace(".png", ".svg"))
    #
    # plt.close(fig)
    # plt.clf()

    for i, df_g_days in enumerate(list_of_df_grouped_by_resolution):
        grouped_by_days = [g for _, g in df_g_days.groupby(['n_days_before_famacha'])]
        plt.clf()
        fig, axs = plt.subplots(2, len(grouped_by_days), figsize=(18., 7.2))
        for j, df_g_days in enumerate(grouped_by_days):
            df_g_days = df_g_days.sort_values(by=['thresh_interpol'])
            dbt = int(df_g_days["n_days_before_famacha"].values[0])
            resolution = df_g_days["resolution"].values[0]
            axs[0, j].plot(df_g_days["thresh_interpol"], df_g_days["usable_11"], label="usable_11")
            axs[0, j].plot(df_g_days["thresh_interpol"], df_g_days["usable_12"], label="usable_12")
            axs[0, j].set_title("%s %d days window" % (resolution, dbt))
            axs[0, j].set(xlabel='Threshold interpolation')
            axs[0, j].legend(loc="upper left")

            df_g_days = df_g_days.sort_values(by=['thresh_zero'])
            dbt = int(df_g_days["n_days_before_famacha"].values[0])
            axs[1, j].plot(df_g_days["thresh_zero"], df_g_days["usable_11"], label="usable_11")
            axs[1, j].plot(df_g_days["thresh_zero"], df_g_days["usable_12"], label="usable_12")
            # axs[1, i].set_title("days before test=%d" % dbt)
            axs[1, j].set(xlabel='Threshold zeros')
            axs[1, j].legend(loc="upper left")

        out_filename = "%s/png/%s_allthresh_%s.png" % (output_dir, resolution, farm_id)
        fig.savefig(out_filename)
        fig.savefig(out_filename.replace(".png", ".svg").replace("png", "svg"))
        print("final file output=", out_filename)
        # plt.show()
        plt.close(fig)
        plt.clf()

    print("finished!")

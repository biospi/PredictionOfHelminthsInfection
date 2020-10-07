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

    print("dataset_folder=", dataset_folder)
    files = glob2.glob(dataset_folder)
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
    df = pd.DataFrame(columns=['usable_11', 'usable_12', 'n_days_before_famacha', 'thresh_interpol', 'thresh_zero'])
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
            except Exception as e:
		print("split=", split)
                print("farm_id=", farm_id)
                print("usable_11=", usable_11)
                print("usable_12=", usable_12)
                print("n_days_before_famacha=", n_days_before_famacha)
                print("thresh_interpol=", thresh_interpol)
                print("thresh_zero=", thresh_zero)
            df.loc[i] = [usable_11, usable_12, n_days_before_famacha, thresh_interpol, thresh_zero]

    # todo reomove
    df_dummy1 = df.copy()
    df_dummy1["n_days_before_famacha"] = 5
    df_dummy_2 = df.copy()
    df_dummy_2["n_days_before_famacha"] = 2
    df_dummy = pd.concat([df, df_dummy1, df_dummy_2])
    df = df_dummy

    print("DF")
    print(df)
    list_of_df = [g for _, g in df.groupby(['n_days_before_famacha'])]

    try:
        print("mkdir=", output_dir)
        pathlib.Path(output_dir).mkdir(parents=True)
    except Exception as e:
        print(e)

    fig, axs = plt.subplots(1, len(list_of_df), figsize=(18., 7.2))
    for i, data_frame in enumerate(list_of_df):
        data_frame = data_frame.sort_values(by=['thresh_zero'])
        dbt = int(data_frame["n_days_before_famacha"].values[0])
        axs[i].plot(data_frame["thresh_zero"], data_frame["usable_11"], label="usable_11")
        axs[i].plot(data_frame["thresh_zero"], data_frame["usable_12"], label="usable_12")
        axs[i].set_title("days before test=%d" % dbt)
        axs[i].set(xlabel='Threshold zeros')
        axs[i].legend(loc="upper left")
    # plt.show()
    out_filename = "%s/thresh_zero_%s_threshi_%d_thresh_z%d.png" % (output_dir, farm_id, thresh_interpol, thresh_zero)
    fig.savefig(out_filename)
    fig.savefig(out_filename.replace(".png", ".svg"))

    plt.close(fig)
    plt.clf()
    fig, axs = plt.subplots(1, len(list_of_df), figsize=(18., 7.2))
    for i, data_frame in enumerate(list_of_df):
        data_frame = data_frame.sort_values(by=['thresh_interpol'])
        dbt = int(data_frame["n_days_before_famacha"].values[0])
        axs[i].plot(data_frame["thresh_interpol"], data_frame["usable_11"], label="usable_11")
        axs[i].plot(data_frame["thresh_interpol"], data_frame["usable_12"], label="usable_12")
        axs[i].set_title("days before test=%d" % dbt)
        axs[i].set(xlabel='Threshold interpolation')
        axs[i].legend(loc="upper left")
    # plt.show()
    out_filename = "%s/thresh_interpol_%s_threshi_%d_thresh_z%d.png" % (output_dir, farm_id, thresh_interpol, thresh_zero)
    fig.savefig(out_filename)
    fig.savefig(out_filename.replace(".png", ".svg"))

    plt.close(fig)
    plt.clf()

    fig, axs = plt.subplots(2, len(list_of_df), figsize=(18., 7.2))
    for i, data_frame in enumerate(list_of_df):
        data_frame = data_frame.sort_values(by=['thresh_interpol'])
        dbt = int(data_frame["n_days_before_famacha"].values[0])
        axs[0, i].plot(data_frame["thresh_interpol"], data_frame["usable_11"], label="usable_11")
        axs[0, i].plot(data_frame["thresh_interpol"], data_frame["usable_12"], label="usable_12")
        axs[0, i].set_title("days before test=%d" % dbt)
        axs[0, i].set(xlabel='Threshold interpolation')
        axs[0, i].legend(loc="upper left")

        data_frame = data_frame.sort_values(by=['thresh_zero'])
        dbt = int(data_frame["n_days_before_famacha"].values[0])
        axs[1, i].plot(data_frame["thresh_zero"], data_frame["usable_11"], label="usable_11")
        axs[1, i].plot(data_frame["thresh_zero"], data_frame["usable_12"], label="usable_12")
        #axs[1, i].set_title("days before test=%d" % dbt)
        axs[1, i].set(xlabel='Threshold zeros')
        axs[1, i].legend(loc="upper left")

    # plt.show()
    plt.close(fig)
    plt.clf()
    out_filename = "%s/allthresh_%s_threshi_%d_thresh_z%d.png" % (output_dir, farm_id, thresh_interpol, thresh_zero)
    fig.savefig(out_filename)
    fig.savefig(out_filename.replace(".png", ".svg"))
    print("final file output=", out_filename)
    print("finished!")










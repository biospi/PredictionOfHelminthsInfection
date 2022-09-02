import json
import statistics
from pathlib import Path
import pandas as pd


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def main(dataset=Path("E:/thesis/datasets/delmas/raw_all_famacha_test/activity_farmid_dbft_7_1min.csv")):
    print(dataset)
    df = pd.read_csv(dataset, header=None)
    cols = list(df.columns)
    cols[-1] = "date"
    cols[-2] = "imputed_days"
    cols[-3] = "id"
    cols[-4] = "label"
    df.columns = cols
    n_animal = df["id"].unique().size
    print(f"n_animal={n_animal}")
    print(df['label'].value_counts(dropna=False))
    print(f"total={df['label'].value_counts(dropna=False).sum()}")

    date_data = pd.to_datetime(df['date'], format='%d/%m/%Y')

    print(f"{date_data.min().strftime('%B %Y')} to {date_data.max().strftime('%B %Y')}")

    print(f"{date_data[0].strftime('%B %Y')}, {date_data[int(date_data.size/2)].strftime('%B %Y')}, "
          f"{date_data[int(date_data.size-1)].strftime('%B %Y')}")


if __name__ == '__main__':
    main(Path("E:/thesis/datasets/delmas/datasetmrnn7_19/activity_farmid_dbft_7_1min.csv"))
    main(Path("E:/Data2/debug3/cedara/dataset6_mrnn_7day/activity_farmid_dbft_7_1min.csv"))
    # delmas_file = "E:/thesis/datasets/delmas/datasetmrnn7_19/activity_farmid_dbft_7_1min.json"
    # cedara_file = "E:/Data2/debug3/cedara/dataset6_mrnn_7day/activity_farmid_dbft_7_1min.json"
    # cedara_weigh_list = [43.9, 45.0, 46.7,50.2,50.4,48.4,46.6,46.3,43.4,40.4,40.5,40.3,39.0,41.9,39.4,40.2,40.5,40.6,40.7,41.9,41.7,44.5,44.3,43.9,43.9,44.0,45.8,46.9,48.0,46.9,48.1,48.2,49.8,49.2,50.7,42.9,43.3,42.0,40.7,41.1,41.9,43.1,43.2]
    # mean_weight = statistics.mean(cedara_weigh_list)
    # print("cedara mean weight=%f" % mean_weight)
    #
    # print("start")
    # for file in [delmas_file, cedara_file]:
    #     with open(file) as fp:
    #         data_famacha_dict = json.load(fp)
    #         cpt_12 = 0
    #         cpt_11 = 0
    #         cpt_21 = 0
    #         cpt_22 = 0
    #         cpt_3 = 0
    #         weights = []
    #         for key, value in data_famacha_dict.items():
    #             for i in range(len(value) - 1):
    #                 curr = value[i]
    #                 next = value[i+1]
    #
    #                 if is_number(curr[3]):
    #                     weights.append(curr[3])
    #
    #
    #                 date1 = curr[0].split('/')
    #                 date2 = next[0].split('/')
    #
    #                 if 'delmas' in file:
    #                     if int(date2[1]) > 4 and int(date2[2]) == 2016:
    #                         continue
    #                     if int(date1[1]) < 3 and int(date2[2]) == 2015:
    #                         continue
    #
    #                 if 'cedara' in file:
    #                     if int(date2[1]) > 7 and int(date2[2]) == 2013:
    #                         continue
    #                     if int(date1[1]) < 4 and int(date2[2]) == 2012:
    #                         continue
    #
    #                 famacha1 = curr[1]
    #                 famacha2 = next[1]
    #                 try:
    #                     int(famacha2)
    #                     int(famacha1)
    #                 except ValueError as e:
    #                     continue
    #                 if famacha1 == famacha2 == 1:
    #                     cpt_11 = cpt_11 + 1
    #                 if famacha1 == famacha2 == 2:
    #                     cpt_22 = cpt_22 + 1
    #                 if famacha1 == 1 and famacha2 == 2:
    #                     cpt_12 = cpt_12 + 1
    #                 if famacha1 == 2 and famacha2 == 1:
    #                     cpt_21 = cpt_21 + 1
    #                 if famacha1 >= 3:
    #                     cpt_3 = cpt_3 + 1
    #                 if famacha2 >= 3:
    #                     cpt_3 = cpt_3 + 1
    #         mean_weight = statistics.mean(weights)
    #         print("mean weight=%f" % mean_weight)
    #         print("cpt_11=%d cpt_22=%d cpt_12=%d cpt_21=%d cpt_3=%d" % (cpt_11, cpt_22, cpt_12, cpt_21, cpt_3))


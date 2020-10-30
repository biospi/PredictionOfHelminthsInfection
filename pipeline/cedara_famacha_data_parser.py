import numpy as np
import pandas as pd
from datetime import datetime
import math
import os
import json
from numbers import Number
from sys import exit
import sys
import datetime as dt
from commands.Herd import AnimalData, HerdFile

if __name__ == '__main__':
    print("Usage: "
          "cedara_famacha_data_parser <output_filename> <raw_famacha_csv_filename>")
    if len(sys.argv) > 1:
        output_filename = sys.argv[1]
        raw_famacha_csv_filename = sys.argv[2]
    else:
        exit(-1)

    out_dir = '/'.join(output_filename.split('/')[:-1])
    print("out_dir=", out_dir)
    if not os.path.exists(out_dir):
        print("mkdir", out_dir)
        os.makedirs(out_dir)

    print("start...")
    # raw_famacha_csv_filename = "E:\\SouthAfrica\\Metadata\\CEDARA data\\Grafieke\\Cedara grafieke.xls"
    print("raw_famacha_csv_filename", raw_famacha_csv_filename)



    df_CStot = pd.read_excel(raw_famacha_csv_filename, skiprows=range(0, 2), sheet_name='CS tot', header=None)
    df_CStot = df_CStot[:-3] #remove bottom 3 rows
    df_CStot = df_CStot.drop(df_CStot.columns[0], axis=1)
    df_CStot = df_CStot.dropna(how='all', axis=1)

    df_MASStot = pd.read_excel(raw_famacha_csv_filename, skiprows=range(0, 2), sheet_name='MASS tot', header=None)
    df_MASStot = df_MASStot[:-3] #remove bottom 3 rows
    df_MASStot = df_MASStot.drop(df_MASStot.columns[0], axis=1)
    df_MASStot = df_MASStot.dropna(how='all', axis=1)

    df_FCtot = pd.read_excel(raw_famacha_csv_filename, skiprows=range(0, 2), sheet_name='FC tot', header=None)
    df_FCtot = df_FCtot[:-3] #remove bottom 3 rows
    df_FCtot = df_FCtot.drop(df_FCtot.columns[0], axis=1)
    df_FCtot = df_FCtot.dropna(how='all', axis=1)

    dfs = np.split(df_FCtot, [df_FCtot.iloc[0].to_list().index('Goat ID') + 1], axis=1)
    df_meta = dfs[0]
    df_fam_data = dfs[1]
    df_mass_data = np.split(df_MASStot, [df_MASStot.iloc[0].to_list().index('Goat ID') + 1], axis=1)[1]
    df_cs_data = np.split(df_CStot, [df_CStot.iloc[0].to_list().index('Goat ID') + 1], axis=1)[1]

    # df_meta = df_meta.rename(columns=df_meta.iloc[0]).drop(df_meta.index[0])
    # df_data = df_data.rename(columns=df_data.iloc[0]).drop(df_data.index[0])

    date_fam = []
    date_mass = []
    date_cs = []
    header_meta = []

    idx_swap = 0
    swap_date = datetime(2013, 2, 14, 0, 0)  # todo Sensor swap date check!!!
    results = []
    for (i, row_meta), (_, row_fam), (_, row_cs), (_, row_mass) in zip(df_meta.iterrows(), df_fam_data.iterrows(),
                                                                       df_cs_data.iterrows(), df_mass_data.iterrows()):
        row_meta_values = [row_meta[col] for col in df_meta.columns]
        f = [row_fam[col] for col in df_fam_data.columns] #get datapoint for the same timestamp
        cs = [row_cs[col] for col in df_cs_data.columns]
        w = [row_mass[col] for col in df_mass_data.columns]

        if i == 0: #header contains date
            header_meta = row_meta_values
            date_fam = f
            date_cs = cs
            date_mass = w
            # Get valid times for famacha and other data that was measured
            time_f = [x.strftime('%d/%m/%Y') for x in np.array(date_fam)]
            itime_f = np.array([int(dt.datetime.strptime(i, '%d/%m/%Y').timestamp()) for i in time_f])

            time_cs = [x.strftime('%d/%m/%Y') for x in np.array(date_cs)]
            itime_cs = np.array([int(dt.datetime.strptime(i, '%d/%m/%Y').timestamp()) for i in time_cs])

            time_w = [x.strftime('%d/%m/%Y') for x in np.array(date_mass)]
            itime_w = np.array([int(dt.datetime.strptime(i, '%d/%m/%Y').timestamp()) for i in time_w])
            continue

        for n in range(len(date_fam)):
            date = date_fam[n]
            if date >= swap_date:
                idx_swap = n
                break

        if i > 75:
            animal_id = row_meta_values[1]
            ftmp = np.array([time_f, itime_f, f], dtype=object)
            cstmp = np.array([time_cs, itime_cs, cs], dtype=object)
            wtmp = np.array([time_w, itime_w, w], dtype=object)
            results.append([animal_id, ftmp, cstmp, wtmp])
        else:
            animal_id = row_meta_values[0]
            ftmp  = np.array([time_f[0:n], itime_f[0:n], f[0:n]], dtype=object)
            cstmp = np.array([time_cs[0:n], itime_cs[0:n], cs[0:n]], dtype=object)
            wtmp  = np.array([time_w[0:n], itime_w[0:n], w[0:n]], dtype=object)
            results.append([animal_id, ftmp, cstmp, wtmp])

            animal_id_ = row_meta_values[1]
            ftmp_  = np.array([time_f[n:], itime_f[n:], f[n:]], dtype=object)
            cstmp_ = np.array([time_cs[n:], itime_cs[n:], cs[n:]], dtype=object)
            wtmp_  = np.array([time_w[n:], itime_w[n:], w[n:]], dtype=object)
            results.append([animal_id_, ftmp_, cstmp_, wtmp_])

    ids = []
    animals = []
    animals_ = {}
    data = {}
    for res in results:
        try:
            animal_id = int(res[0])
        except ValueError as e:
            print(e)
            continue

        ftmp = res[1]
        cstmp = res[2]
        wtmp = res[3]

        df_ftmp = pd.DataFrame(ftmp)
        df_cstmp = pd.DataFrame(cstmp)
        df_wtmp = pd.DataFrame(wtmp)

        #filter out values if data is str or missing
        df_ftmp = df_ftmp.loc[:, [(type(x) != str) for x in df_ftmp.iloc[2, :]]]
        df_cstmp = df_cstmp.loc[:, [(type(x) != str) for x in df_cstmp.iloc[2, :]]]
        df_wtmp = df_wtmp.loc[:, [(type(x) != str) for x in df_wtmp.iloc[2, :]]]

        df_ftmp = df_ftmp.dropna(axis='columns')
        df_cstmp = df_cstmp.dropna(axis='columns')
        df_wtmp = df_wtmp.dropna(axis='columns')

        if animal_id not in ids:
            animals_[animal_id] = {"data": [df_ftmp, df_cstmp, df_wtmp]}
            ids.append(animal_id)
        else:
            animals_[animal_id]["swaped"] = [df_ftmp, df_cstmp, df_wtmp]

    for key in animals_.keys():
        if "swaped" in animals_[key]:
            df_ftmp = pd.concat([animals_[key]["data"][0], animals_[key]["swaped"][0]], axis=1)
            df_cstmp = pd.concat([animals_[key]["data"][1], animals_[key]["swaped"][1]], axis=1)
            df_wtmp = pd.concat([animals_[key]["data"][2], animals_[key]["swaped"][2]], axis=1)
            animals.append(AnimalData(key, df_ftmp.values, df_cstmp.values, df_wtmp.values))
        else:
            df_ftmp = animals_[key]["data"][0]
            df_cstmp = animals_[key]["data"][1]
            df_wtmp = animals_[key]["data"][2]
            animals.append(AnimalData(key, df_ftmp.values, df_cstmp.values, df_wtmp.values))

    print(data)
    print('finished')
    print('dumping result to json file...')
    # Save Herd data to HDF5 File
    hfile = HerdFile(output_filename)
    hfile.saveHerd(animals)


    # old_keys = data.keys()
    # data_formated = {}
    # for key in old_keys:
    #     new_key = "4001130" + str(key).zfill(4) #full id is missing from raw spreadsheet
    #     for item in data[key]:
    #         item[2] = new_key
    #     data_formated[new_key] = data[key]
    # print(data_formated)
    # with open(output_filename, 'w') as fp:
    #     json.dump(data_formated, fp)
    # print(output_filename)

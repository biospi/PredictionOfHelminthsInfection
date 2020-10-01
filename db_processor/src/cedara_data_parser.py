import xlrd
import numpy as np
import pandas as pd
from datetime import datetime
import math
import os
import json
from numbers import Number
from sys import exit
from ipython_genutils.py3compat import xrange
pd.set_option('display.max_columns', 48)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 1000)

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))


def excel_date_2_datetime(excel_date):
    return datetime(*xlrd.xldate_as_tuple(excel_date, 0))


if __name__ == '__main__':
    print("start...")
    path = "E:\\SouthAfrica\\Metadata\\CEDARA data\\Grafieke\\Cedara grafieke.xls"
    print(path)
    data = {}
    swap_date = datetime(2013, 2, 14, 0, 0)
    df_list = [pd.read_excel(path, skiprows=range(0, 2), sheet_name='FC tot', header=None),
               pd.read_excel(path, skiprows=range(0, 2), sheet_name='FC drench', header=None),
               pd.read_excel(path, skiprows=range(0, 2), sheet_name='FC monitor', header=None)]
    for df in df_list:
        df = df[:-3]
        df = df.drop(df.columns[0], axis=1)
        df = df.dropna(how='all', axis=1)
        dfs = np.split(df, [df.iloc[0].to_list().index('Goat ID')+1], axis=1)
        df_meta = dfs[0]
        df_data = dfs[1]

        # df_meta = df_meta.rename(columns=df_meta.iloc[0]).drop(df_meta.index[0])
        # df_data = df_data.rename(columns=df_data.iloc[0]).drop(df_data.index[0])

        # print(df_meta)
        header_meta = []
        header_data = []
        for (i, row_meta), (j, row_data) in zip(df_meta.iterrows(), df_data.iterrows()):
            row_meta_values = [row_meta[col] for col in df_meta.columns]
            row_data_values = [row_data[col] for col in df_data.columns]

            if i == 0:
                header_meta = row_meta_values
                header_data = row_data_values
                continue

            # print(i, row_meta_values, row_data_values)

            for n, famacha in enumerate(row_data_values):
                if famacha == np.nan or famacha == '*' or (isinstance(famacha, Number) and math.isnan(famacha)):
                    continue

                date = header_data[n]
                i_d = 1 if date >= swap_date else 0
                id = row_meta_values[i_d]

                if id == np.nan or id == '*' or (isinstance(id, Number) and math.isnan(id)):
                    continue

                if id in data:
                    data[id].append([date.strftime('%d/%m/%Y'), famacha, id, 0])
                else:
                    data[id] = [[date.strftime('%d/%m/%Y'), famacha, id, 0]]
    print(data)
    print('finished')
    print('dumping result to json file...')

    old_keys = data.keys()
    data_formated = {}
    for key in old_keys:
        new_key = "4001130" + str(key).zfill(4)
        for item in data[key]:
            item[2] = new_key
        data_formated[new_key] = data[key]
    print(data_formated)
    farm_name = "cedara"
    with open(__location__+'\\%s_famacha_data.json' % farm_name, 'w') as fp:
        print('')
        json.dump(data_formated, fp)


        # break

    # book = xlrd.open_workbook(path)
    # for n in range(book.nsheets):
    #     sheet = book.sheet_by_index(n)
    #
    #     if 'FC' not in sheet.name:
    #         continue
    #
    #     for row_index in xrange(0, sheet.nrows):
    #         try:
    #             row_values = [sheet.cell(row_index, col_index).value for col_index in xrange(0, sheet.ncols)]
    #             print(row_values)
    #         except Exception as e:
    #             print(e)

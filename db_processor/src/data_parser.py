import time
import os
import xlrd
from datetime import datetime
from ipython_genutils.py3compat import xrange
import json
from sys import exit
import pandas as pd

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))


def purge_file(filename):
    print("purge %s..." % filename)
    try:
        os.remove(filename)
    except FileNotFoundError:
        print("file not found.")


def find_data_files(directory_path, extension='.xls'):
    print("start searching for data files...")
    os.chdir(directory_path)
    file_paths = [val for sublist in
                  [[os.path.join(i[0], j) for j in i[2] if j.endswith(extension)] for i in os.walk(directory_path)] for
                  val in sublist]
    print("founded %d %s files" % (len(file_paths), extension))
    return file_paths


def find_sheet_containing_valid_famacha(book, farm_name, path):
    #returns index of famacha column if exists
    map = {}
    for n in range(book.nsheets):
        sheet = book.sheet_by_index(n)
        for row_index in xrange(0, sheet.nrows):
            try:
                row_values = [sheet.cell(row_index, col_index).value for col_index in xrange(0, sheet.ncols)]

                if 'bothaville' in farm_name:
                    if "Neus" in row_values and "Kwak keel" in row_values and "Sender nr" in row_values:
                        map[sheet.name] = {"id_col_index": row_values.index("Sender nr"), "famacha_col_index": row_values.index("Kwak keel")-1}
                        continue

                    if "Neus" in row_values and "Kwak keel" in row_values and "Sk/Bk ID" in row_values:
                        map[sheet.name] = {"id_col_index": row_values.index("Sk/Bk ID"), "famacha_col_index": row_values.index("Kwak keel")-1}

                if 'cedara' in farm_name:

                    if 'FC' not in sheet.name:
                        continue

                    if "Transponder" in row_values:
                        map[path] = {"id_col_index": row_values.index("Transponder"),
                                           "famacha_col_index": row_values.index("Fc")}
            except Exception as e:
                print(e)
    return map


def format_date(date):
    split = date.split('-')
    # print("%s-%s-%s" % (split[0], split[1], split[2][-2:]))
    return "%s-%s-%s" % (split[0], split[1], split[2][-2:])


def build_famacha_data(map, book, data):
    df = pd.read_csv("E:/SouthAfrica/Metadata/BOTHAVILLE data/Bothaville Sender nr.csv", sep=",")
    for key, value in map.items():
        sheet = book.sheet_by_name(key)
        for row_index in xrange(0, sheet.nrows):
            row_values = [sheet.cell(row_index, col_index).value for col_index in xrange(0, sheet.ncols)]
            try:

                id = str(row_values[value['id_col_index']]).split('.')[0]
                id_map = df.set_index('Sk ID').T.to_dict('list')
                id_map = {str(key): "40121100" + str(value[0]).split('.')[0] if len(str(value[0]).split('.')[0]) == 3 else "4012110" + str(value[0]).split('.')[0] for key, value in id_map.items()}
                # id_map = {str(key): int(value) for key, value in id_map.items()}

                if id in id_map:
                    id = id_map[id]

                if len(id) < 10:
                    continue
                id = int(id)
                famacha = int(row_values[value['famacha_col_index']])
                date = datetime.strptime(format_date(key), '%d-%m-%y').strftime('%d/%m/%Y')
            except (TypeError, ValueError) as e:
                print(e)
                continue
            print(id, famacha, date)
            if id in data.keys():
                if [date, famacha, id, 0] in data[int(id)]:
                    continue
                data[int(id)].append([date, famacha, id, 0])
            else:
                data[int(id)] = [[date, famacha, id, 0]]

    data = {int(key): value for key, value in data.items()}
    return data


def process_files(file_paths, farm_name=''):
    data = {}
    for curr_file, path in enumerate(file_paths):
        print("reading file...")
        print(path)
        book = xlrd.open_workbook(path)
        map = find_sheet_containing_valid_famacha(book, farm_name, path)
        if not map:
            continue
        print(map)
        data = build_famacha_data(map, book, data)
    print('dumping result to json file...')
    print(data)
    filepath = __location__+'\\%s_famacha_data_with_age.json' % farm_name
    with open(filepath, 'w') as fp:
        print(filepath)
        json.dump(data, fp)


if __name__ == '__main__':
    print("start...")
    start_time = time.time()
    # xls_files = find_data_files("E:/SouthAfrica/Metadata/CEDARA2 data", extension='.xls')
    xls_files = find_data_files("E:/SouthAfrica/Metadata/BOTHAVILLE data", extension='.xls')
    process_files(xls_files, farm_name="bothaville")

    # pdf_files = find_data_files("E:/SouthAfrica/Metadata/BOTHAVILLE data", extension='.pdf')

    exit()


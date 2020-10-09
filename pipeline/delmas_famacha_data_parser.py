import json
import os
import sys
from sys import exit

import xlrd


def generate_table_from_xlsx(path):
    book = xlrd.open_workbook(path)
    sheet = book.sheet_by_index(1)
    data = {}
    for row_index in range(0, sheet.nrows):
        row_values = [sheet.cell(row_index, col_index).value for col_index in range(0, sheet.ncols)]
        if row_index == 2:
            time_w = list(filter(None, row_values))
            # print(time_w)

        if row_index == 3:
            idx_w = [index for index, value in enumerate(row_values) if value == "WEIGHT"]
            idx_c = [index for index, value in enumerate(row_values) if value == "FAMACHA"]

        chunks = []
        if row_index > 4:
            for i in range(0, len(idx_w)):
                if row_values[1] is '':
                    continue
                s = "40101310%s" % row_values[1]
                serial = int(s.split('.')[0])
                chunks.append([time_w[i], row_values[idx_c[i]], serial, row_values[idx_w[i]]])
            if len(chunks) != 0:
                data[serial] = chunks
    # print(data)
    return data


if __name__ == '__main__':
    print("args: output_filename raw_famacha_csv_filename")
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
    data_famacha_dict = generate_table_from_xlsx(raw_famacha_csv_filename)
    print("data_famacha_dict=", data_famacha_dict)
    with open(output_filename, 'a') as outfile:
        json.dump(data_famacha_dict, outfile)
    print("export file=", output_filename)
    print("done")



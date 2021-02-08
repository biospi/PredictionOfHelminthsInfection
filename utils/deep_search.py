import json
from pathlib import Path
import pandas as pd
import os

import tables
import xlrd
import h5py as h5

if __name__ == "__main__":

    # path = "F:/Data2/h5/cedara.h5"
    # h5_raw = tables.open_file(path, "r")
    # data = h5_raw.root.table
    # list_id_raw = {}
    # for idx, x in enumerate(data):
    #     if x['first_sensor_value'] in list_id_raw:
    #         list_id_raw[x['first_sensor_value']] = list_id_raw[x['first_sensor_value']] + 1
    #     else:
    #         list_id_raw[x['first_sensor_value']] = 1

    directory_path = "F:/SouthAfrica/Tracking Data/Cedara"
    os.chdir(directory_path)
    file_paths = [val for sublist in
                  [[os.path.join(i[0], j) for j in i[2] if j.endswith('.xlsx')] for i in os.walk(directory_path)] for
                  val in sublist]
    file_paths = [x.replace("\\", "/") for x in file_paths]

    # missing_cedara = [1006, 1015, 1021, 1024, 1028, 1035, 1037, 1042, 1055, 1068, 1074, 1102, 1115, 1129, 1135, 1503, 1506, 1509, 1511,
    #  1521, 1524, 1529, 1532, 1539, 1542, 1545, 1555, 1559, 1560, 1565, 1571, 1573, 1575, 1577, 1579, 1581, 1585, 1591,
    #  1594, 1595, 1599, 461, 468, 578, 604, 617, 618, 622, 625, 628, 631, 634, 649, 651, 657, 661, 665, 666, 668, 674,
    #  675, 676, 677, 680, 684, 685, 695, 697, 698, 700, 708, 715, 729, 734, 744, 747, 749, 750, 752, 754, 755, 756, 759,
    #  762, 767, 774, 783, 784, 789, 804, 807, 808, 809, 815, 816, 820, 827, 831, 833, 855, 858, 863, 865, 869, 875, 883,
    #  885, 886]

    missing_cedara = [461, 468, 578, 593, 604, 617, 618, 622, 625, 628, 631, 634, 649, 651, 657, 661, 665, 666, 668, 674, 675, 676, 677,
     680, 684, 685, 695, 697, 698, 700, 708, 715, 729, 734, 744, 747, 749, 750, 752, 754, 755, 756, 759, 762, 767, 774,
     783, 784, 789, 804, 807, 808, 809, 815, 816, 820, 827, 831, 833, 855, 858, 863, 869, 875, 885, 886]


    report = {}
    for i, path in enumerate(file_paths):
        if '~$' in path:
            continue
        print("%d/%d" % (i, len(file_paths)), path)
        try:
            book = xlrd.open_workbook(path)
        except Exception as e:
            print(e)
            continue
        for sheet in book.sheets():
            for row_index in range(0, sheet.nrows):
                row_values = str([sheet.cell(row_index, col_index).value for col_index in range(0, sheet.ncols)])
                # print(row_values)
                for missing in missing_cedara:
                    if str(missing) in row_values:
                        # print("FOUND MISSING DATA !!!")
                        # print(missing)
                        # print(path)
                        filen = path
                        if missing not in report:
                            report[missing] = {}

                        if filen not in report[missing]:
                            report[missing] = {filen: 1}
                        else:
                            report[missing][filen] = report[missing][filen] + 1
                        break

    print(report)
    filename = "F:/Data2/deep_search_report.json"
    json.dump(report, open(filename, 'w'))

    # dataDir = Path(path)
    # files = sorted(dataDir.glob("*.csv"))
    # files = [[x, int(x.stem)] for x in files]
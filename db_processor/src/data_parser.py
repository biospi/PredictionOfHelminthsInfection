import time
import os
import xlrd
from ipython_genutils.py3compat import xrange


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


def find_files_containing_valid_famacha(book):
    #returns index of famacha column if exists
    map = {}
    for n in range(book.nsheets):
        sheet = book.sheet_by_index(n)
        for row_index in xrange(0, sheet.nrows):
            try:
                row_values = [sheet.cell(row_index, col_index).value for col_index in xrange(0, sheet.ncols)]
                if "Neus" in row_values and "Kwak keel" in row_values and "Sender nr" in row_values:
                    map[sheet.name] = {"id_col_index": row_values.index("Sender nr"), "famacha_col_index": row_values.index("Kwak keel")-1}
            except Exception as e:
                print(e)
    return map


def process_files(file_paths):
    valid_files = []
    for curr_file, path in enumerate(file_paths):
        try:
            print("reading file...")
            print(path)
            book = xlrd.open_workbook(path)
            map = find_files_containing_valid_famacha(book)
            print(map)
            # if not is_valid:
            #     continue
            # valid_files.append(path)
            # for n in range(book.nsheets):
            #     sheet = book.sheet_by_index(n)
            #     found_col_index = False
            #     for row_index in xrange(0, sheet.nrows):
            #         try:
            #             row_values = [sheet.cell(row_index, col_index).value for col_index in xrange(0, sheet.ncols)]
            #             if "Sender nr" not in row_values:
            #                 continue
            #
            #             famacha_value = row_values[f_col_index]
            #             id = row_values[row_values.index("Sender nr")]
            #             date = sheet.name
            #
            #             print(row_values)
            #         except Exception as e:
            #             print(e)
        except Exception as e:
            print(e)

    print("found %d valid files out of %d." % (len(valid_files), len(file_paths)) )


if __name__ == '__main__':
    print("start...")
    start_time = time.time()
    xls_files = find_data_files("E:/SouthAfrica/Metadata/BOTHAVILLE data", extension='.xls')
    pdf_files = find_data_files("E:/SouthAfrica/Metadata/BOTHAVILLE data", extension='.pdf')

    process_files(xls_files)


import math
import os

"""
Utility class for static methods
"""

def anscombe(value):
    try:
        return 2 * math.sqrt(value + (3 / 8))
    except TypeError as e:
        print(e)

def create_rec_dir(path):
    dir_path = ""
    sub_dirs = path.split("/")
    for sub_dir in sub_dirs[0:]:
        dir_path += sub_dir + "/"
        # print("sub_folder=", dir_path)
        if not os.path.exists(dir_path):
            print("mkdir", dir_path)
            os.makedirs(dir_path)
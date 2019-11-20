import os
os.environ[
    "PROJ_LIB"] = "C:\\Users\\fo18103\\AppData\\Local\\Continuum\\anaconda3\\pkgs\\proj4-5.2.0-ha925a31_1\\Library\\share"
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from datetime import datetime
import traceback
import imageio


def images_to_gif(filenames):
    print(filenames)
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave('anim.gif', images)


def draw_frame(lons, lats, altitude, dates, id):
    fig, axs = plt.subplots(2, 1)
    fig.set_size_inches(10, 15)
    fig.suptitle(str(dates[id-1]), fontsize=20, x=0.5, y=0.97)

    axs[0].set_title("  ")
    m = Basemap(width=1000, height=1000,  projection='merc',
                ax=axs[0],
                llcrnrlat=-27.47, urcrnrlat=-27.43,
                llcrnrlon=29.9, urcrnrlon=29.95,
                lat_ts=1,
                resolution='f')
    m.drawmapboundary()
    m.drawcoastlines(color='black', linewidth=0.4)
    m.drawrivers(color='blue')
    m.fillcontinents(color='lightgray')
    x, y = m(lons, lats)  # transform coordinates
    m.plot(x, y, 'ro-', markersize=3, linewidth=1)

    axs[1].set_title('altitude')
    axs[1].plot(dates, altitude)
    fig.show()
    fig.savefig(str(id), dpi=100)
    return "%d.png" % id


if __name__ == '__main__':
    f = open("E:\\Kraal\\Kraal Tag Data (GAF54 02 to 04 SEP19) 377rows.csv", 'r')

    lats = []
    lons = []
    altitude = []
    dates = []
    image_paths = []
    lines = f.readlines()
    f.close()

    for n, line in enumerate(lines):
        if n <= 0:
            continue

        try:
            print('.', end='')
            splitLine = line.split(',')
            date = datetime.strptime(splitLine[27].strip(), '%Y-%m-%d %H:%M:%S')
            st = splitLine[13].strip()
            st = st.strip("'")
            altitude.append(float(st))

            lat = splitLine[8].strip()
            lat = lat.strip("'")
            lats.append(float(lat))

            lng = splitLine[9].strip()
            lng = lng.strip("'")
            lons.append(float(lng))

            dates.append(date)

            path = draw_frame(lons, lats, altitude, dates, n)
            image_paths.append(path)
        except Exception as e:
            print("Error", e)
            print(traceback.format_exc())
            print(line)

    images_to_gif(image_paths)



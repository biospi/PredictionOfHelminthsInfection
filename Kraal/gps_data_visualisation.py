import os
os.environ[
    "PROJ_LIB"] = "C:\\Users\\fo18103\\AppData\\Local\\Continuum\\anaconda3\\pkgs\\proj4-5.2.0-ha925a31_1\\Library\\share"
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from datetime import datetime
import traceback
import imageio
import os.path
import glob
from sys import exit
from colour import Color


def uniqueish_color():
    """There're better ways to generate unique colors, but this isn't awful."""
    return plt.cm.gist_ncar(np.random.random())


def images_to_gif_():
    filenames = []
    # os.chdir("/mydir")
    for file in glob.glob("*.png"):
        v = int(file.split('.')[0])
        file_n = "%05d.png" % v
        filenames.append(file_n)
        # os.rename(os.path.join(file), os.path.join(file_n))

    filenames.sort()
    print(filenames)
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave('anim.gif', images)
    exit()


def images_to_gif(filenames):
    filenames.sort()
    print(filenames)
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave('anim.gif', images)


def draw_frame(lons, lats, altitudes, curr_date, dates, id, m, fig, axs):
    chuncks_lats = []
    chuncks_lons = []
    elems_lats = []
    elems_lons = []
    cpt = 1
    for n in range(len(lons)):
        elems_lats.append(lats[n])
        elems_lons.append(lons[n])

        if cpt == 2:
            cpt = 0
            chuncks_lats.append(elems_lats)
            elems_lats = []
            chuncks_lons.append(elems_lons)
            elems_lons = []
        cpt += 1

    if len(chuncks_lats) < len(lats):
        if len(chuncks_lats) == 0:
            chuncks_lats.append(lats[-1])
            chuncks_lons.append(lons[-1])
        else:
            chuncks_lats.append([chuncks_lats[-1][-1], lats[-1]])
            chuncks_lons.append([chuncks_lons[-1][-1], lons[-1]])

    fig.suptitle(str(curr_date), fontsize=20, x=0.5, y=0.97)

    colors = list(Color("red").range_to(Color("green"), len(chuncks_lats)))

    for i in range(len(chuncks_lats)):
        x, y = m(chuncks_lons[i], chuncks_lats[i])  # transform coordinates
        # print(colors[i].rgb)
        m.plot(x, y, 'o-', markersize=3, linewidth=1, color=colors[i].rgb)

    # fig.suptitle(str(curr_date), fontsize=20, x=0.5, y=0.97)
    # x, y = m(lons, lats)
    # m.plot(x, y, 'ro-', markersize=3, linewidth=1)
    axs[1].set_title('altitude')
    axs[1].plot(dates, altitudes, 'bo-')
    #fig.show()
    path = "%05d.png" % id
    fig.savefig(path, dpi=100)
    return path


def init_basemap():
    fig, axs = plt.subplots(2, 1)
    fig.set_size_inches(10, 15)
    axs[0].set_title("  ")
    m = Basemap(width=1000, height=1000,  projection='merc',
                ax=axs[0],
                llcrnrlat=-27.46, urcrnrlat=-27.439,
                llcrnrlon=29.929, urcrnrlon=29.945,
                lat_ts=1,
                resolution='f')
    m.drawmapboundary()
    m.drawcoastlines(color='black', linewidth=0.4)
    m.drawrivers(color='blue')
    m.fillcontinents(color='darkgray')
    return fig, axs, m


if __name__ == '__main__':
    # images_to_gif_()
    f = open("E:\\Kraal\\Kraal Tag Data (GAF54 02 to 04 SEP19) 377rows.csv", 'r')

    lats = []
    lons = []
    altitude = []
    dates = []
    image_paths = []
    lines = f.readlines()
    lines.reverse()
    f.close()

    fig, axs, m = init_basemap()
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

            path = draw_frame(lons, lats, altitude, date, dates, n, m, fig, axs,)
            image_paths.append(path)
        except Exception as e:
            print("Error", e)
            print(traceback.format_exc())
            print(line)

    images_to_gif(image_paths)



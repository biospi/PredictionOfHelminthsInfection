#
# Author: Ranjeet Bhamber <ranjeet <a.t> bristol.ac.uk>
#
# Copyright (C) 2020  Biospi Laboratory for Medical Bioinformatics, University of Bristol, UK
#
# This file is part of PredictionOfHelminthsInfection.
#
# PHI is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PHI is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with seaMass.  If not, see <http://www.gnu.org/licenses/>.
#
import h5py as h5
import numpy as np
from pathlib import Path


class AnimalData:
    ID = 0
    Tag = 0
    famacha = []
    csScore = []
    weight = []
    def __init__(self, _tag, _famacha, _cs, _weight):
        self.Tag = _tag
        self.famacha = _famacha
        self.csScore = _cs
        self.weight = _weight

    def addData(self, _famacha, _cs, _weight):
        self.famacha = _famacha
        self.csScore = _cs
        self.weight = _weight

    def appendData(self, _famacha, _cs, _weight):
        self.famacha.append(_famacha)
        self.csScore.append(_cs)
        self.weight.append(_weight)

    def TagToIDSet(self, tag):
        if self.Tag in tag[1, :]:
            # Method 1
            # match element in numpy array and give out just the matching Correspond virtically aligned ID, and not a 1
            # element numpy array
            # x[0, x[1,:] == 110][0]
            # To get just index
            # np.nonzero( x[1,:] == 110 )
            # self.ID = tag[0, tag[1, :] == self.Tag][0]
            # Method 2
            # match elemnt in array or matrix and given index is in a Tuple and strip from np array to just element
            # idx = np.where(x[1,:] == 110)[0][0]
            # idx = np.where(tag[1,:] == self.Tag)[0][0]
            idx = np.where(tag[1, :] == self.Tag)[0][0]
            self.ID = tag[0, idx]
        else:
            print("Error cannot find Tag in activity Animal List!")
            print(f"Cannot find ID for animal Tag: {self.Tag}")


class herd_data:
    herd = []
    missing = 0

    def __init__(self):
        self.herd = []
        self.missing = 0

    def addHerd(self, _herd):
        self.herd = _herd

    def set_id_of_herd(self, tagList):
        for idxh in self.herd:
            idxh.TagToIDSet(tagList)
        self.missing = [(idx, x.Tag) for idx, x in enumerate(self.herd) if x.ID == 0]

    def get_missing_list(self):
        for idx in self.missing:
            print(f"Missing Tag Number: {idx[1]} \t Index: {idx[0]}")
        return self.missing

    def remove_missing(self):
        print("Deleting the following missing (Idx, Tags)", self.missing)
        offset = 0
        for idx in self.missing:
            print(f"Deleting Tag: {idx[1]} from Index: {idx[0]}")
            del self.herd[idx[0]-offset]
            offset += 1


    def getAnimalIDList(self):
        return [x.ID for x in self.herd]


class herd_file:
    herdFile = []

    def __init__(self, fileName):
        self.herdFile = Path(fileName)

    def saveHerd(self, herd):
        # open hdf5 file;
        fh5 = h5.File(self.herdFile, 'w')

        for idx in herd:
            gd = fh5.create_group(str(idx.Tag))
            # Famacha
            ft = idx.famacha[1, :].astype(int)
            fd = idx.famacha[2, :].astype(int)
            # Conditioning Score
            ct = idx.csScore[1, :].astype(int)
            cd = idx.csScore[2, :].astype(float)
            # Weight
            wt = idx.weight[1, :].astype(int)
            wd = idx.weight[2, :].astype(float)

            gd.create_dataset('famachaTime', data=ft, compression="gzip")
            gd.create_dataset('famacha', data=fd, compression="gzip")

            gd.create_dataset('csTime', data=ct, compression="gzip")
            gd.create_dataset('cs', data=cd, compression="gzip")

            gd.create_dataset('weightTime', data=wt, compression="gzip")
            gd.create_dataset('weight', data=wd, compression="gzip")

        fh5.close()

    def loadHerd(self):
        herdData = []

        ahf = h5.File(self.herdFile, 'r')
        animal = list(ahf.keys())

        for i in animal:
            Tag = int(i)
            gdata = ahf[i]

            t = np.array(gdata['csTime'])
            x = np.array(gdata['cs'])
            cs = np.array([t, x])

            t = np.array(gdata['famachaTime'])
            x = np.array(gdata['famacha'])
            famacha = np.array([t, x])

            t = np.array(gdata['weightTime'])
            x = np.array(gdata['weight'])
            weight = np.array([t,x])
            herdData.append(AnimalData(Tag, famacha, cs, weight))
        ahf.close()
        return herdData

    def load_herd(self, _herd):
        herdData = []

        ahf = h5.File(self.herdFile, 'r')
        animal = list(ahf.keys())

        for i in animal:
            Tag = int(i)
            gdata = ahf[i]

            t = np.array(gdata['csTime'])
            x = np.array(gdata['cs'])
            cs = np.array([t, x])

            t = np.array(gdata['famachaTime'])
            x = np.array(gdata['famacha'])
            famacha = np.array([t, x])

            t = np.array(gdata['weightTime'])
            x = np.array(gdata['weight'])
            weight = np.array([t,x])
            herdData.append(AnimalData(Tag, famacha, cs, weight))
        ahf.close()
        _herd.addHerd(herdData)


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
    famacha = []
    csScore = []
    weight = []
    def __init__(self, _id, _famacha, _cs, _weight):
        self.ID = _id
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


class HerdFile:
    herdFile = []
    def __init__(self, fileName):
        self.herdFile = Path(fileName)

    def saveHerd(self, herd):
        # open hdf5 file;
        fh5 = h5.File(self.herdFile, 'w')

        for idx in herd:
            gd = fh5.create_group(str(idx.ID))
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
            ID = int(i)
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
            herdData.append(AnimalData(ID, famacha, cs, weight))
        ahf.close()

        return herdData




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

# %%
import numpy as np
import pandas as pd
# from enum import Enum, auto
try:
    from typing import Final
except ImportError:
    from typing_extensions import Final
from pathlib import Path
import datetime as dt
import math

from dataset.cmdsextra import bc, vbprintSet

vbprint = vbprintSet(True)

class TimeConst:
    utc1min: Final = 60
    utc1hour: Final = 60 * utc1min
    utc1day: Final = 24 * utc1hour


# class FamachaVal(Enum):
#    f1 = auto()
#    f2 = auto()
#    f3 = auto()
#    f4 = auto()
#    f5 = auto()
#    f1To1 = auto()
#    f1To2 = auto()
#    f1To3 = auto()
#    f2To2 = auto()
#    f2To1 = auto()
#    f2To3 = auto()
#    f3To3 = auto()
#    f3To2 = auto()
#    f3To1 = auto()
#    def getFamacha(self, famachVal):
#        if famachVal =

# More cleaver way than to use enumerations to do famacha fast and can cover all combinations.
class FamachaVal:
    f = {
        0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 10: '10',
        11: '1To1', 12: '1To2', 13: '1To3', 14: '1To4', 15: '1To5',
        21: '2To1', 22: '2To2', 23: '2To3', 24: '2To4', 25: '2To5',
        31: '3To1', 32: '3To2', 33: '3To3', 34: '3To4', 35: '3To5',
        41: '4To1', 42: '4To2', 43: '4To3', 44: '4To4', 45: '4To5',
        51: '5To1', 52: '5To2', 53: '5To3', 54: '5To4', 55: '5To5'
    }

    def getFamachaV(self, fInt):
        return (fInt, self.f[fInt])

    def getFamachaS(self, fStr):
        return (list(self.f.keys())[list(self.f.values()).index(fStr)], fStr)

    def getDeltaFamacha(self, fA, fB):
        df = fA[0] * 10 + fB[0]
        return self.getFamachaV(df)


class activity_file:
    dataDir = 0
    files = 0
    ID = 0
    col = None

    def __init__(self, _dataDir, col):
        self.col = col
        self.dataDir = Path(_dataDir)
        self.files = sorted(self.dataDir.glob('*.csv'))
        _ID = [x.stem[0:x.stem.find('_')] if '_' in x.stem else x.stem for x in self.files]
        # Get rid of any median file, this will be unnecessary in the future
        if 'median' in _ID:
            idx = _ID.index('median')
            del self.files[idx]
            del _ID[idx]

        # x = np.array([int(i) for i in _ID])
        # mask = int(x[0] / 1000) * 1000
        # self.ID = np.array([x, x - mask])
        x = np.array([int(i) for i in _ID])
        x_sub = np.array([int(str(i)[-4:]) for i in _ID])
        self.ID = np.array([x, x_sub])

    def getFileNames(self):
        return self.files

    def getID(self):
        return self.ID

    def loadActivityTraceList(self, _ID):
        """
        Load in data from directory containing csv files containing activity trace data return list of Activity
        Trace objects.
        """
        animalTrace = []
        for id in _ID:
            idx = np.where(self.ID == id)[1][0]
            print("Loading Activity from file: ", self.files[idx])
            dataFrame = pd.read_csv(self.files[idx])

            # xmin = dataFrame.loc[:, 'xmin']
            # xmax = dataFrame.loc[:, 'xmax']
            # ymin = dataFrame.loc[:, 'ymin']
            # ymax = dataFrame.loc[:, 'ymax']
            # zmin = dataFrame.loc[:, 'zmin']
            # zmax = dataFrame.loc[:, 'zmax']
            #
            # magnitude_min = math.sqrt(xmin * xmin + ymin * ymin + zmin * zmin)
            # magnitude_max = math.sqrt(xmax * xmax + ymax * ymax + zmax * zmax)

            atrace = np.array(dataFrame.loc[:, self.col])
            atime = np.array(dataFrame.loc[:, 'timestamp'])
            animalTrace.append(Activity(self.ID[0, idx], atrace, atime))
        return animalTrace

    def loadActivityTrace(self, _ID, tag=None):
        """
        Load in data from directory containing csv files containing activity trace data return Single Activity
        Trace object.
        """

        idx = np.where(self.ID == _ID)[1][0]

        print(f"Loading Activity from file: {bc.CYAN}{self.files[idx]}{bc.ENDC}")
        dataFrame = pd.read_csv(self.files[idx])

        # if _ID == 99999999999: #todo remove this is just for heatmap visualisation no need to save empty transponder trace otherwise
        #     filepath = self.files[idx].parent / f"{tag}.csv"
        #     print(filepath)
        #     dataFrame_ = dataFrame.copy()
        #     dataFrame.to_csv(filepath, index=False)

        atrace = np.array(dataFrame.loc[:, self.col])
        missingness = np.zeros(dataFrame.shape[0])
        if 'imputed' in dataFrame.columns:
            missingness = np.array(dataFrame.loc[:, 'imputed'])

        # xmin = dataFrame.loc[:, 'xmin']
        # xmax = dataFrame.loc[:, 'xmax']
        # ymin = dataFrame.loc[:, 'ymin']
        # ymax = dataFrame.loc[:, 'ymax']
        # zmin = dataFrame.loc[:, 'zmin']
        # zmax = dataFrame.loc[:, 'zmax']
        #
        # magnitude_min = np.sqrt(xmin * xmin + ymin * ymin + zmin * zmin)
        # magnitude_max = np.sqrt(xmax * xmax + ymax * ymax + zmax * zmax)
        #
        # atrace = np.array(magnitude_max)

        atime = np.array(dataFrame.loc[:, 'timestamp'])
        return Activity(self.ID[0, idx], atrace, atime), Activity(self.ID[0, idx], missingness, atime)


class Activity:
    ID = []  # ID of Animal
    T = []  # Time of each activity measurement
    A = []  # Activity trace of animal

    def __init__(self, _id, _trace, _time):
        self.ID = _id
        self.A = _trace
        self.T = _time

    def getTime(self):
        return np.array([dt.datetime.fromtimestamp(x) for x in self.T])

    def getUTC(self):
        return self.T

    def getTrace(self):
        return self.A


#class Sample:
#    ID = 0
#    itime = 0
#    activity = 0
#    famacha = 0
#    deltaFamacha = 0
#    valid = False
#
#    def __init__(self, _ID, _itime, _act, _fam=0, _df=0):
#        self.ID = _ID
#        self.itime = np.array(_itime)
#        self.activity = np.array(_act)
#        self.famacha = _fam
#        self.deltaFamacha = _df
#        # If there are no nans in the activity then set valid to True
#        self.valid = math.isnan(self.activity.min()) == False

class Sample:
    ID = 0
    famacha = 0
    deltaFamacha = 0
    missRate = 0.0
    valid = False

    def __init__(self, _ID=0,  _fam=0, _df=0, _valid = None, _missRate = 0.0):
        self.ID = _ID
        self.famacha = _fam
        self.deltaFamacha = _df
        # If there are no nans in the activity then set valid to True
        self.valid = _valid
        self.missRate = _missRate


class SampleSet:
    set = []
    rawA = []
    rawT = []
    rawID = []
    N = 0
    iT = 0
    iA = 0
    valid = 0
    df = 0

    def __init__(self):
        self.set = []
        self.rawA = []
        self.rawT = []
        self.rawID = []
        self.N = 0
        self.iT = 0
        self.iA = 0
        self.valid = 0
        self.df = 0

    def generateSet(self, famData, actFile, deltaFamachaTime):
        N = len(famData.herd)
        deltaTime = deltaFamachaTime * TimeConst.utc1day
        ftool = FamachaVal()

        fdt = dt.datetime.fromtimestamp
        for i, aIdx in enumerate(famData.herd):
            try:
                act, miss = actFile.loadActivityTrace(aIdx.ID)
            except IndexError as e:
                print(e)
                print("missing transponder data!")
                continue
                # act, miss = actFile.loadActivityTrace(99999999999, tag=aIdx.Tag)
                # aIdx.ID = aIdx.Tag
                # act.ID = aIdx.Tag

            print(f"Procesing animal with ID:{bc.BLUE} {act.ID} \t [{i+1}/{N}] {bc.ENDC}")
            aniSet = []
            T0 = act.T[0]
            dT = act.T[1] - act.T[0]
            TE = act.T[act.T.shape[0]-1]

            self.rawA.append(act.T)
            self.rawT.append(act.A)
            self.rawID.append(act.ID)

            for j in np.r_[1:aIdx.famacha.shape[1]]:
                vbprint(f"Procesing Famacha: {bc.MAG} [{j}/{aIdx.famacha.shape[1]}] {bc.ENDC}")
                fTimeIdxEnd = aIdx.famacha[0, j]
                fTimeIdxStart = self.getStartIndex(fTimeIdxEnd, deltaTime)
                vbprint(f"UTC : Start Time of Famacha: {fTimeIdxStart} \t End Time of Famacha: {fTimeIdxEnd}")
                vbprint(f"HR  : Start Time of Famacha: {dt.datetime.fromtimestamp(fTimeIdxStart)} \t End Time of Famacha: {dt.datetime.fromtimestamp(fTimeIdxEnd)}")

                fCurrent = ftool.getFamachaV(aIdx.famacha[1, j])
                fPrev = ftool.getFamachaV(aIdx.famacha[1, j - 1])
                df = ftool.getDeltaFamacha(fPrev, fCurrent)
                vbprint(f"Previous Famacha: {fPrev[1]} \t Current Famacha: {fCurrent[1]} \t Delta Famaacha: {df}")

                if fTimeIdxStart > TE or fTimeIdxEnd > TE:
                    vbprint(f"{bc.YELLOW}Start Time of Famacha {bc.RED}{fdt(fTimeIdxStart)}{bc.YELLOW} is greater than Time for activity {bc.RED}{fdt(TE)}{bc.ENDC}")
                    vbprint(f"or")
                    vbprint(f"{bc.YELLOW}End Time of Famacha {bc.RED}{fdt(fTimeIdxEnd)}{bc.YELLOW} is greater than Time for activity {bc.RED}{fdt(TE)}{bc.ENDC}")
                    # break

                TIdxStart = int((fTimeIdxStart - T0) / dT)
                TIdxEnd = int((fTimeIdxEnd - T0) / dT)

                if TIdxStart < 0:
                    print("outside of activity range", dt.datetime.fromtimestamp(fTimeIdxStart))
                    continue

                # if fTimeIdxStart < T0:
                #     #vbprint(f"{bc.YELLOW}Start Time of Famacha {fdt(TIdxStart)} is less than Time for activity {fdt(T0)}{bc.ENDC}")
                #     continue

                T = np.array(act.T[TIdxStart:TIdxEnd+1])
                A = np.array(act.A[TIdxStart:TIdxEnd+1])
                M = np.array(miss.A[TIdxStart:TIdxEnd+1])

                w_size = (deltaFamachaTime * 1440)+1
                if A.size != w_size:
                    # date_range = np.array(pd.date_range(dt.datetime.fromtimestamp(fTimeIdxStart), periods=w_size, freq='T').tolist())
                    # epoch_range = [int(dt.datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S').timestamp()) for x in date_range]
                    # print("edit", w_size, A.size, A)
                    # T = epoch_range
                    # A = np.zeros(len(epoch_range))
                    #
                    # A[:] = np.nan
                    continue

                # Is activity valid
                # val = math.isnan(A.min()) == False
                # val = np.isnan(A).tolist().count(True) < (720) * deltaFamachaTime
                #missR = np.sum(1 - M) / M.size  # missing marked as 0
                valid_imputed_day = 0

                range_ = np.arange(0, M.shape[0], 1440)
                for k in range(len(range_)-1):
                    d_ = M[range_[k]:range_[k+1]]
                    if np.all(d_ == 1):
                        valid_imputed_day += 1

                # cpt = 0
                # for elem in M:
                #     if elem > 0:
                #         cpt += 1
                #     if cpt == 1440-1:
                #         valid_imputed_day += 1
                #         cpt = 0

                val = valid_imputed_day >= 1

                if len(A[np.isnan(A)]) == len(A):
                    val = False
                    #valid_imputed_day = 0


                aniSet.append(Sample(aIdx.ID, fCurrent, df, val, valid_imputed_day))

                if type(self.iA) == int:
                    self.iA = np.array(A)
                    self.iT = np.array(T)
                else:
                    self.iA = np.vstack((self.iA, np.array(A)))
                    self.iT = np.vstack((self.iT, np.array(T)))

            self.set.extend(aniSet)

        self.valid = np.array([x.valid for x in self.set])
        self.df = np.array([x.deltaFamacha[1] for x in self.set])

    def getStartIndex(self, famTimeEnd, deltaTime):
        famTimeStart = famTimeEnd - deltaTime
        return famTimeStart

    def getSet(self):
        return self.set

    def getActivity(self, idx = 0):
        if idx == 0:
            return self.iA
        else:
            return self.iA[idx]

    def getiTime(self, idx = 0):
        if idx == 0:
            return self.iT
        else:
            return self.iT[idx]

    def getFamachaCase(self, fcase = 'all'):
        if fcase == 'all':
            return self.df
        else:
            return np.where(self.df == fcase)

    def getValid(self, state = True):
        if state:
            return self.valid
        else:
            return self.valid == False


# class Famacha:
#    class Score:
#        def __init__(self, _date, _famacha, _weight):
#            self.date = _date
#            self.famacha = _famacha
#            self.weight = _weight
#    date = 0
#    dateInt = 0
#    famacha = 0
#    weight = 0
#    ID = 0
#    famachaList = []
#
#    def __init__(self, _id):
#        ID = _id
#
#    def addScore(self, _date, _famacha, _weight):
#        self.famachaList.append(self.Score(_date, _famacha, _weight))
#
#    def getFamachaList(self):
#        return self.famachaList
#


if __name__ == "__main__":
    print("Test this module code should go here...")

#    import sys
#    from Herd import HerdFile, HerdData
#
#    if len(sys.argv) != 4:
#        print("Usage: "
#              "createSample <Famacha HDF5 file> <Directory with data> <Output Directory>")
#        exit(1)
#
#    famFile = Path(sys.argv[1])
#    dataDir = Path(sys.argv[2])
#    outDir = Path(sys.argv[3])
#
#    print("Commanline argument: ", famFile)
#
#    fileHerd = HerdFile(famFile)
#    famachaHerd = HerdData()
#
#    # load Famacha based scores from HDF5
#    fileHerd.loadHerd(famachaHerd)
#
#
#    missheep = fileHerd.getf
#
#    print("Loading Activity traces and Times of famacha based animals")
#    aData = ActivityFile(dataDir)
#
#    sheepID = aData.getID()
#
#
#    [x.TagToIDSet(sheepID) for x in famachaHerd]
#
#
#    #Load only data based on Famacha data.
#    aT = aData.loadActivityTrace(goatID)
#
#    print(aT)
#

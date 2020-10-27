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
from typing import Final
from pathlib import Path
import datetime as dt
import math
import h5py as h5

from commands.cmdsextra import bc, vbprintSet

vbprint = vbprintSet(False)

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
        1: '1', 2: '2', 3: '3', 4: '4', 5: '5',
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


class ActivityFile:
    dataDir = 0
    files = 0
    ID = 0

    def __init__(self, _dataDir):
        self.dataDir = Path(_dataDir)
        self.files = sorted(self.dataDir.glob('*.csv'))
        _ID = [x.stem[0:x.stem.find('_')] for x in self.files]
        # Get rid of any median file, this will be unnecessary in the future
        if 'median' in _ID:
            idx = _ID.index('median')
            del self.files[idx]
            del _ID[idx]

        x = np.array([int(i) for i in _ID])
        mask = int(x[0] / 1000) * 1000
        self.ID = np.array([x, x - mask])

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
            atrace = np.array(dataFrame.loc[:, 'first_sensor_value'])
            atime = np.array(dataFrame.loc[:, 'timestamp'])
            animalTrace.append(Activity(self.ID[0, idx], atrace, atime))
        return animalTrace

    def loadActivityTrace(self, _ID):
        """
        Load in data from directory containing csv files containing activity trace data return Single Activity
        Trace object.
        """
        idx = np.where(self.ID == _ID)[1][0]
        print(f"Loading Activity from file: {bc.CYAN}{self.files[idx]}{bc.ENDC}")
        dataFrame = pd.read_csv(self.files[idx])
        atrace = np.array(dataFrame.loc[:, 'first_sensor_value'])
        atime = np.array(dataFrame.loc[:, 'timestamp'])
        return Activity(self.ID[0, idx], atrace, atime)


class Activity:
    ID = []  # ID of Animal
    T = []  # Time of each activity measurement
    A = []  # Activity trace of animal

    def __init__(self, _id, _trace, _time):
        self.ID = _id
        self.A = _trace
        self.T = _time

    def getTime(self):
        return np.array([dt.datetime.utcfromtimestamp(x) for x in self.T])

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
    valid = False

    def __init__(self, _ID=0,  _fam=0, _df=0, _valid = None):
        self.ID = _ID
        self.famacha = _fam
        self.deltaFamacha = _df
        # If there are no nans in the activity then set valid to True
        self.valid = _valid


def saveSampleSetHDF5(dataSet, fileName = 'sampleDataset.h5'):
    fName = Path(fileName)

    famacha = np.array([x.famacha[0] for x in dataSet.set])
    valid = np.array([1 if x == True else 0 for x in dataSet.valid])
    sampleID = np.array([x.ID for x in dataSet.set])
    df = np.array([FamachaVal.getFamachaS(FamachaVal, i)[0] for i in dataSet.df])
    iA = dataSet.iA
    iT = dataSet.iT

    t0 = max([x[0] for x in dataSet.rawT])
    tN = min(([x[len(x)-1] for x in dataSet.rawT]))

    id = np.array(dataSet.rawID)
    rawA = 0
    rawT = 0
    for i in np.r_[0:len(dataSet.rawT)]:
        idx0 = np.where(dataSet.rawT[i] == t0)[0][0]
        idxN = np.where(dataSet.rawT[i] == tN)[0][0]

        if type(rawA) == int:
            rawA = np.array(dataSet.rawA[i][idx0:idxN])
            rawT = np.array(dataSet.rawT[i][idx0:idxN])
        else:
            rawA = np.vstack((rawA, np.array(dataSet.rawA[i][idx0:idxN])))

    fs = h5.File(fName, 'w')
    print("Writing Data to HDF5 File.")
    fs.create_dataset('famacha',data=famacha.astype(int), compression='gzip')
    fs.create_dataset('valid',data=valid.astype(int), compression='gzip')
    fs.create_dataset('df',data=df.astype(int), compression='gzip')
    fs.create_dataset('sampleID',data=sampleID.astype(int), compression='gzip')
    fs.create_dataset('iA',data=iA.astype(float), compression='gzip')
    fs.create_dataset('iT',data=iT.astype(int) , compression='gzip')
    fs.create_dataset('rawID',data=id.astype(int), compression='gzip')
    fs.create_dataset('rawA',data=rawA.astype(float), compression='gzip')
    fs.create_dataset('rawT',data=rawT.astype(int), compression='gzip')

    fs.close()

def loadSampleSetHDF5(fileName):
    fName = Path(fileName)
    dataSet = SampleSet()

    fs = h5.File(fName, 'r')


    df = np.array(fs['df'])
    df = [FamachaVal.getFamachaV(FamachaVal, x) for x in df]
    famacha = np.array(fs['famacha'])
    famacha = [FamachaVal.getFamachaV(FamachaVal, x) for x in famacha]
    sampleID = np.array(fs['sampleID'])
    valid = np.array(fs['valid'])
    valid = [True if x == 1 else False for x in valid]

    for i in np.r_[0:len(sampleID)]:
        dataSet.set.append(Sample(sampleID[i], famacha[i], df[i], valid[i]))

    iA = np.array(fs['iA'])
    iT = np.array(fs['iT'])
    rawID = np.array(fs['rawID'])
    rawA = np.array(fs['rawA'])
    rawT = np.array(fs['rawT'])

    dataSet.N = len(dataSet.set)
    dataSet.iA = iA
    dataSet.iT = iT
    dataSet.valid = valid
    dataSet.df = df

    dataSet.rawA = rawA
    dataSet.rawT = rawT
    dataSet.rawID = rawID

    fs.close()

    return dataSet


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

    def generateSet(self, famData, actFile, deltaFamachaTime):
        N = len(famData.herd)
        deltaTime = deltaFamachaTime * TimeConst.utc1day
        ftool = FamachaVal()

        fdt = dt.datetime.utcfromtimestamp
        for i, aIdx in enumerate(famData.herd):
            act = actFile.loadActivityTrace(aIdx.ID)
            print(f"Procesing animal with ID:{bc.BLUE} {act.ID} \t [{i+1}/{N}] {bc.ENDC}")
            aniSet = []
            T0 = act.T[0]
            dT = act.T[1] - act.T[0]
            TE = act.T[act.T.shape[0]-1]

            self.rawA.append(act.A)
            self.rawT.append(act.T)
            self.rawID.append(act.ID)

            for j in np.r_[1:aIdx.famacha.shape[1]]:
                vbprint(f"Procesing Famacha: {bc.MAG} [{j}/{aIdx.famacha.shape[1]}] {bc.ENDC}")
                fTimeIdxEnd = aIdx.famacha[0, j]
                fTimeIdxStart = self.getStartIndex(fTimeIdxEnd, deltaTime)
                vbprint(f"UTC : Start Time of Famacha: {fTimeIdxStart} \t End Time of Famacha: {fTimeIdxEnd}")
                vbprint(f"HR  : Start Time of Famacha: {dt.datetime.utcfromtimestamp(fTimeIdxStart)} \t End Time of Famacha: {dt.datetime.utcfromtimestamp(fTimeIdxEnd)}")

                fCurrent = ftool.getFamachaV(aIdx.famacha[1, j])
                fPrev = ftool.getFamachaV(aIdx.famacha[1, j - 1])
                df = ftool.getDeltaFamacha(fPrev, fCurrent)
                vbprint(f"Previous Famacha: {fPrev[1]} \t Current Famacha: {fCurrent[1]} \t Delta Famaacha: {df}")

                if fTimeIdxStart < T0:
                    vbprint(f"{bc.YELLOW}Start Time of Famacha {fdt(TIdxStart)} is less than Time for activity {fdt(T0)}{bc.ENDC}")
                    continue
                if fTimeIdxStart > TE or fTimeIdxEnd > TE:
                    vbprint(f"{bc.YELLOW}Start Time of Famacha {bc.RED}{fdt(fTimeIdxStart)}{bc.YELLOW} is greater than Time for activity {bc.RED}{fdt(TE)}{bc.ENDC}")
                    vbprint(f"or")
                    vbprint(f"{bc.YELLOW}End Time of Famacha {bc.RED}{fdt(fTimeIdxEnd)}{bc.YELLOW} is greater than Time for activity {bc.RED}{fdt(TE)}{bc.ENDC}")
                    break

                TIdxStart = int((fTimeIdxStart - T0) / dT)
                TIdxEnd = int((fTimeIdxEnd - T0) / dT)

                T = np.array(act.T[TIdxStart:TIdxEnd+1])
                A = np.array(act.A[TIdxStart:TIdxEnd+1])

                # Is activity valid
                val = math.isnan(A.min()) == False

                aniSet.append(Sample(aIdx.ID, fCurrent, df, val))

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

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

import numpy as np
import pandas as pd
from enum import Enum, auto
from typing import Final
from pathlib import Path
import datetime as dt

class TimeConst:
    utc1min: Final = 60
    utc1hour: Final = 60*utc1min
    utc1day: Final = 24*utc1hour

class Score(Enum):
    f1 = auto()
    f2 = auto()
    f3 = auto()
    f4 = auto()
    f5 = auto()
    f1To1 = auto()
    f1To2 = auto()
    f1To3 = auto()
    f2To2 = auto()
    f2To1 = auto()
    f2To3 = auto()
    f3To3 = auto()
    f3To2 = auto()
    f3To1 = auto()

class Activity:
    def __init__(self, _id, _trace, _time):
        self.ID = _id
        self.A = _trace
        self.T = _time
    def getTime(self):
        return np.array([dt.datetime.utcfromtimestamp(x) for x in  self.T])
    def getUTC(self):
        return self.T
    def getTrace(self):
        return self.A
    ID = [] # ID of Animal
    T = []  # Time of each activity measurement
    A = []  # Activity trace of animal

class Sample:
    ID = 0
    date = 0
    activity = 0
    famacha = 0
    deltaFamacha = 0

def loadActivityTrace(dirName):
    """
    Load in data from directory containing csv files containing activity trace data return list of Activity
    Trace objects.
    """
    files = Path(dirName)
    fileName = [ x.stem[0:x.stem.find('_')] for x in files]

    animalTrace = []

    # Get rid of any median file, this will be unnecessary in the future
    if 'median' in fileName:
        idx = fileName.index('median')
        del files[idx]
        del fileName[idx]

    for i, idx in enumerate(files):
        dataFrame = pd.read_csv(i)
        atrace = np.array(dataFrame.loc[:,'first_sensor_value'])
        atime = np.array(dataFrame.loc[:,'timestamp'])
        animalTrace.append(Activity(fileName[idx], atrace, atime))
    return animalTrace


#%%

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3
        print("Usage: "
              "createSample <Famacha.json file> <Directory with data> <Output Directory> ")

    josonFile = Path(sys.argv[1])
    dataDir = Path(sys.argv[2])
    outDir = Path(sys.argv[3])

    print("Commanline argument: ", josonFile)

    files = sorted(dataDir.glob("*.csv"))

    for i in files:
        print("list all dir conenst: ", i)

    for

    datetime.utcfromtimestamp(y.timestamp)

    dt.datetime.utcfromtimestamp(y.timestamp)

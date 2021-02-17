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

#%%
import sys

from commands.Herd import *
from commands.Samples import *
from commands.cmdsextra import bc

#%%
if sys.argv == ['']:
    famFile = '..\Data\\delmas_animal_data.h5'
    dataDir = 'H:\\Data\\Axel_Famacha\\data_pipeline\\p3_csv_threshold\\interpol_5_zeros_480'
    dataOut = "C:\\Users\\ranjeet\\Repository\\PycharmProjects\\bunker\\dataOut"
elif len(sys.argv) != 4:
    print("Usage: "
          "createSample <Famacha HDF5 file> <Directory with data> <Output Directory>")
    exit(1)
else:
    famFile = Path(sys.argv[1])
    dataDir = Path(sys.argv[2])
    outDir = Path(sys.argv[3])

#%% Load in data

print("Commanline argument: ", famFile)

print("Loading Famacha and Sorce based data from HDF5 file")
fileHerd = HerdFile(famFile)
famachaHerd = HerdData()

print("Loading Activity traces and Times of famacha based animals")
activityData = ActivityFile(dataDir)

# load Famacha based scores from HDF5
fileHerd.loadHerd(famachaHerd)

# load Filenames of activity data.
sheepID = activityData.getID()

famachaHerd.setIDofHerd(sheepID)

#%%
mis = famachaHerd.getMissingList()

famachaHerd.removeMissing()

# Load only data based on Famacha data.
samples = SampleSet()

samples.generateSet(famachaHerd, activityData, 1)

#aTraces = activityData.loadActivityTraceList(famachaHerd.getAnimalIDList())

#sTraces = activityData.loadActivityTrace(famachaHerd.herd[0].ID)


#totalS = len([ele for sub in samples.set for ele in sub])
#validS = len([ele for sub in samples.set for ele in sub if ele.valid == True])
#falseS = len([ele for sub in samples.set for ele in sub if ele.valid == False])

totalS = len(samples.set)
validS = len([x for x in samples.set if x.valid == True])
falseS = len([x for x in samples.set if x.valid == False])

print(f"{bc.MAG}Summary of extracted samples:{bc.ENDC}")

print(f"Number of Samples extracted: {bc.BLUE}{totalS} {bc.ENDC}")
print(f"Number of Samples extracted: {bc.GREEN}{validS} {bc.ENDC}")
print(f"Number of Samples extracted: {bc.RED}{falseS} {bc.ENDC}")


saveSampleSetHDF5(samples,'testSampleSetNow.h5')

snew = loadSampleSetHDF5('testSampleSetNow.h5')

print(f"Number of Samples extracted: {bc.BLUE}{totalS} {bc.ENDC}")
print(f"Number of Valid ValSamples extracted: {bc.GREEN}{validS} {bc.ENDC}")
print(f"Number of InValid Samples extracted: {bc.RED}{falseS} {bc.ENDC}")

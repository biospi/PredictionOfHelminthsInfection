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

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import random as rd
import h5py as h5
import datetime as dt
#import pickle
from Herd import AnimalData, HerdFile


if len(sys.argv) != 3:
    print("Only will extract weight, CS and Famacha from Excel sheet RAW formated in certain way."
          "Warning when processing Excel files make SURE ALL NUMBERS IN EXCEL ARE CONVERTED TO NUMS AND NOT STRING!!!"
          "Usage: "
          "famacha_delmas.py <Famacha Excel File> <FileOut> ")
    exit(1)
famFile = Path(sys.argv[1])
outFile = Path(sys.argv[2])

# Process this Famacha File.
# Warning when processing Excel files make SURE ALL NUMBERS ARE CONVERTED TO NUMS AND NOT STRING !!!! Bullshit BUG
# famFile = 'D:\\Data\\Axel_Famacha\\data_pipeline\\0_raw_data\\delmas\\famacha_csv\\Famacha_processed.xlsx'

raw = pd.read_excel(famFile, sheet_name='Raw')

# Find all index in time row where we do not have a nan.
timeIdx = raw.iloc[0,:].isnull() == False

tval  = raw.iloc[0,:]

# Get valid times for famacha and other data that was measured
time = tval[timeIdx].to_numpy()

#convert to datetime object and return UTC seconds from 1970,1,1
itime = np.array([int(dt.datetime.strptime(i, '%d/%m/%Y').timestamp()) for i in time])

# Select only animal data:
dfani = raw.iloc[3:,1:]

row, col = dfani.shape

# Python list containing all animals
animals = []

# Index for animal to loop over data.
a_idx = 0
while a_idx < 35:
    print("Processing Animal: ", a_idx)

    # Get animal ID or the last 3 significant digist of it.
    animal_id = dfani.iloc[a_idx, 0]

    # Get all data for weight, cs and famacha
    f  = dfani.iloc[a_idx, 3::3].to_numpy()
    cs = dfani.iloc[a_idx, 2::3].to_numpy()
    w  = dfani.iloc[a_idx, 1::3].to_numpy()

    # Filter data and only get Index of numbers in arrays. Filter out any strings and filter out any nans!
    fidx  = [idx for idx, x in enumerate(f)  if type(x) != str if np.isnan(x) == False]
    csidx = [idx for idx, x in enumerate(cs) if type(x) != str if np.isnan(x) == False]
    widx  = [idx for idx, x in enumerate(w)  if type(x) != str if np.isnan(x) == False]


    # Make temp timestamp and data measurements tuple list version:
    # ftmp  = list(zip(time[fidx],  f[fidx]))
    # cstmp = list(zip(time[csidx], cs[csidx]))
    # wtmp  = list(zip(time[widx],  w[widx]))

    # Make temp timestamp and data measurements numpy array version:
    ftmp  = np.array([time[fidx],  itime[fidx],  f[fidx]])
    cstmp = np.array([time[csidx], itime[csidx], cs[csidx]])
    wtmp  = np.array([time[widx],  itime[widx],  w[widx]])

    # Add all valid data including time stamps to list of all animals
    animals.append(AnimalData(animal_id, ftmp, cstmp, wtmp ))

    a_idx += 1


#print(animals)

print("Random test of values parsed from file Check with origonal!!!")
for i in range(0,10):
    print("Test ", i)
    atest = rd.choice(animals)

    print("Animal ID: ", atest.ID)
    r, c = atest.weight.shape
    tidx = rd.randint(0,c-1)
    ftest  = atest.famacha[:, tidx]
    cstest = atest.csScore[:, tidx]
    wtest  = atest.weight[:, tidx]

    #    r, c =   atest.famacha.shape
    #    ftest  = atest.famacha[:, rd.randint(0,c-1)]
    #    r, c =   atest.csScore.shape
    #    cstest = atest.csScore[:, rd.randint(0,c-1)]
    #    r, c =   atest.weight.shape
    #    wtest  = atest.weight[:, rd.randint(0,c-1)]

    print("Famacha test: ", ftest)
    print("csScore test: ", cstest)
    print("Weight  test: ", wtest)

# Save Herd data to HDF5 File
hfile = HerdFile(outFile)
hfile.saveHerd(animals)

#print("Saving data to python data format using pickle:")
#
#with open('famacha_data_python.data', 'wb') as filehandle:
#    # store the data as binary data stream
#    pickle.dump(animals, filehandle)




import pandas as pd
import numpy as np
from pathlib import Path
import sys
import random as rd
import pickle

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



#if len(sys.argv) != 2:
#    print("Only will extract weight, CS and Famacha from Excel sheet RAW formated in certain way."
#          "Warning when processing Excel files make SURE ALL NUMBERS ARE CONVERTED TO NUMS AND NOT STRING !!!!"
#          "Usage: "
#          "Famacha_delmas.py <Famacha Excel File>")
#    exit(1)
#famFile = Path(sys.argv[1])

# Process this Famacha File.
# Warning when processing Excel files make SURE ALL NUMBERS ARE CONVERTED TO NUMS AND NOT STRING !!!! Bullshit BUG
famFile = 'D:\\Data\\Axel_Famacha\\data_pipeline\\0_raw_data\\delmas\\famacha_csv\\Famacha_processed.xlsx'

raw = pd.read_excel(famFile, sheet_name='Raw')

# Find all index in time row where we do not have a nan.
timeIdx = raw.iloc[0,:].isnull() == False

tval  = raw.iloc[0,:]

# Get valid times for famacha and other data that was measured
time = tval[timeIdx].to_numpy()

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
    ftmp  = np.array([time[fidx],  f[fidx]])
    cstmp = np.array([time[csidx], cs[csidx]])
    wtmp  = np.array([time[widx],  w[widx]])


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


print("Saving data to python data format using pickle:")

with open('famacha_data_python.data', 'wb') as filehandle:
    # store the data as binary data stream
    pickle.dump(animals, filehandle)


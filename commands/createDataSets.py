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
import os

from commands.Herd import *
from commands.Samples import *
from commands.cmdsextra import bc
from utils.Utils import create_rec_dir

if len(sys.argv) != 6:
    print("Usage: "
          "createSample <Famacha HDF5 file> <Directory with data> <Output Directory> <Data column> <Days>")
    exit(1)

famFile = Path(sys.argv[1])
dataDir = Path(sys.argv[2])
outDir = Path(sys.argv[3])
#first_sensor_value_gain
#first_sensor_value
#first_sensor_value_li
data_col = sys.argv[4]
ndays = int(sys.argv[5])

print("Commanline argument: ", famFile)


print("Loading Famacha and Sorce based data from HDF5 file")
fileHerd = HerdFile(famFile)
famachaHerd = HerdData()

print("Loading Activity traces and Times of famacha based animals")
activityData = ActivityFile(dataDir, data_col)

# load Famacha based scores from HDF5
fileHerd.loadHerd(famachaHerd)

# load Filenames of activity data.
sheepID = activityData.getID()

famachaHerd.setIDofHerd(sheepID)

mis = famachaHerd.getMissingList()

famachaHerd.removeMissing()

# Load only data based on Famacha data.
samples = SampleSet()

samples.generateSet(famachaHerd, activityData, ndays)



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
print(f"Number of Valid Samples extracted: {bc.GREEN}{validS} {bc.ENDC}")
print(f"Number of NaN Samples extracted: {bc.RED}{falseS} {bc.ENDC}")

# Get all meta data for each sample im our generated data set

metaSet = np.array(samples.getSet())
actSet = samples.getActivity()
timeSet = samples.getiTime()

set1To2 = samples.getFamachaCase("1To2")
idx = samples.getFamachaCase("1To2")
set1To2M = metaSet[idx]
set1To2A = actSet[idx]
set1To2T = timeSet[idx]


# targets_info = {"total": {"all": totalS}}
targets_info = {"total": {"all": totalS, "valid": np.where(samples.valid == True)[0].size, "not_valid": np.where(samples.valid == False)[0].size}}
for target in list(set(samples.df)):
    targets_info[target] = {"all": np.where(samples.df == target)[0].size, "valid": 0, "not_valid": 0}

# targets_info = {"total": {"valid": np.where(samples.valid == True)[0].size, "not_valid": np.where(samples.valid == False)[0].size, "all": totalS}}
# for target in list(set(samples.df)):
#     targets_info[target] = {"total": np.where(samples.df == target)[0].size, "valid": 0, "not_valid": 0}

for i in np.where(samples.valid == False)[0]:
    for k in targets_info.keys():
        if samples.df[i] == k:
            targets_info[k]["not_valid"] += 1

for i in np.where(samples.valid == True)[0]:
    for k in targets_info.keys():
        if samples.df[i] == k:
            targets_info[k]["valid"] += 1
#
# for k in targets_info.keys():
#     targets_info[k]["all"] = targets_info[k]["not_valid"] + targets_info[k]["valid"]

split = dataDir.name.split("_")
import datetime
import json
# farm = "delmas"
# base_station = 70101200027

# farm = "cedara"
# base_station = 70091100056
create_rec_dir(str(outDir))
# farm_id = str(dataDir).split('\\')[-1]
farm_id = "delmas_70101200027"
filename = "%s/activity_%s_dbft_%d_1min.json" % (outDir, farm_id, ndays)
json.dump(targets_info, open(filename, 'w'))

s = []
#meta = [0, '02/12/2015', '01/12/2015', 40101310050, 2, 1, 1, 1, -1, '02/12/2015', '15/12/2015', '08/01/2016', '15/01/2016', '22/01/2016', 13, 24, 31, 7]
#valid_idx = np.where(samples.valid == True)[0]
for idx in range(totalS):
    # meta[1] = datetime.datetime.fromtimestamp(samples.iT[idx][-1]).strftime('%d/%m/%Y')
    # meta[2] = datetime.datetime.fromtimestamp(samples.iT[idx][0]).strftime('%d/%m/%Y')
    # meta[9] = datetime.datetime.fromtimestamp(samples.iT[idx][-1]).strftime('%d/%m/%Y')
    # meta[3] = samples.set[idx].ID
    sample = samples.iA[idx].tolist() + [samples.df[idx]] + [samples.set[idx].ID, samples.set[idx].missRate, datetime.datetime.fromtimestamp(samples.iT[idx][-1]).strftime('%d/%m/%Y')]
    s.append(sample)
df = pd.DataFrame(s)
df.to_csv(filename.replace(".json", ".csv"), sep=',', index=False, header=False)




# total_sample_11 = np.where(samples.df == '1To1')[0].size
# total_sample_12 = np.where(samples.df == '1To2')[0].size
#
# nan_sample_11 = 0
# nan_sample_12 = 0
# for i in np.where(samples.valid == False)[0]:
#     if samples.df[i] == '1To1':
#         nan_sample_11 += 1
#     if samples.df[i] == '1To2':
#         nan_sample_12 += 1
#
# usable_11 = 0
# usable_12 = 0
# for i in np.where(samples.valid == True)[0]:
#     if samples.df[i] == '1To1':
#         usable_11 += 1
#     if samples.df[i] == '1To2':
#         usable_12 += 1
#
# split = dataDir.name.split("_")
# import datetime
# report = "Total samples = %d\n1 -> 1 = %d\n1 -> 2 = %d\nNan samples: \n1 -> 1 = %d\n1 -> 2 = %d\nUsable: \n1 " \
#          "-> 1 = %d\n1 -> 2 = %d\n" % (
#          totalS, total_sample_11, total_sample_12, nan_sample_11,
#          nan_sample_12, usable_11, usable_12)
#
# filename = "F:/Data2/gen_dataset_new/activity_delmas_70101200027_dbft_%d_1min_threshi_%d_threshz_%d.txt" % (ndays, int(split[3]), int(split[5]))
# with open(filename, 'a') as outfile:
#     outfile.write(report)
#     outfile.write('\n')
#     outfile.close()
#
# s = []
# meta = [1442, '02/12/2015', '01/12/2015', 40101310050, 2, 1, 1, 1, -1, '02/12/2015', '15/12/2015', '08/01/2016', '15/01/2016', '22/01/2016', 13, 24, 31, 7]
# valid_idx = np.where(samples.valid == True)[0]
# for idx in valid_idx:
#     if samples.df[idx] == '1To1':
#         target = "False"
#     if samples.df[idx] == '1To2':
#         target = "True"
#     if samples.df[idx] not in ['1To1', '1To2']:
#         continue
#
#     meta[1] = datetime.datetime.fromtimestamp(samples.iT[idx][-1]).strftime('%d/%m/%Y')
#     meta[2] = datetime.datetime.fromtimestamp(samples.iT[idx][0]).strftime('%d/%m/%Y')
#     meta[9] = datetime.datetime.fromtimestamp(samples.iT[idx][-1]).strftime('%d/%m/%Y')
#     meta[3] = samples.set[idx].ID
#     sample = samples.iA[idx].tolist() + [target] + meta
#     s.append(sample)
#
#     # sample = samples.iA[idx].tolist() + ["median_"+target] + meta
#     # s.append(sample)
#     #
#     # sample = samples.iA[idx].tolist() + ["mean_"+target] + meta
#     # s.append(sample)
#
#
# df = pd.DataFrame(s)
# df.to_csv("F:/Data2/gen_dataset_new/activity_delmas_70101200027_dbft_%d_1min_threshi_%d_threshz_%d.csv" % (ndays, int(split[3]), int(split[5])), sep=',', index=False, header=False)
#

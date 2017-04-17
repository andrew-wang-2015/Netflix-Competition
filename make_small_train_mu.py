import pandas as pd
import numpy as np

idxFile  = '../mu/all.idx'
dataFile = '../mu/all.dta'

indices = pd.read_csv(idxFile, delimiter=' ', header=None, nrows=1000).values
print('Done loading all.idx - 1000')
data = pd.read_csv(dataFile, delimiter=' ' , header=None, nrows=1000).values
print('Done loading all.dta - 1000')

trainIndices = np.where(indices <= 3)[0]
print('Done finding train indices')

trainData = data[trainIndices, :]
print('Created trainData')

df1 = pd.DataFrame(trainData)
print('Created df1')

df1.to_csv('../mu/small_train.dta', index=False, sep=' ', header=False)
print('Created small_train.dta')
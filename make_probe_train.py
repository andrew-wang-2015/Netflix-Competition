import numpy as np
import pandas as pd

idxFile  = 'um/all.idx'
dataFile = 'um/all.dta'

def main():
	indices = pd.read_csv(idxFile, delimiter=' ', header=None).values
	print('Done loading all.idx')
	data = pd.read_csv(dataFile, delimiter=' ' , header=None).values
	print('Done loading all.dta')

	probeIndices = np.where(indices == 4)[0]
	print('Done finding probe indices')
	trainIndices = np.where(indices <= 3)[0]
	print('Done finding train indices')

	trainData = data[trainIndices, :]
	print('Created trainData')
	probeData = data[probeIndices, :]
	print('Created probeData')

	df1 = pd.DataFrame(trainData)
	print('Created df1')
	df2 = pd.DataFrame(probeData)
	print('Created df2')

	df1.to_csv('train.dta', sep=' ', index=False, header=False)
	print('Created train.dta')
	df2.to_csv('probe.dta', sep=' ', index=False, header=False)
	print('Created probe.dta')

if __name__ == '__main__':
	main()
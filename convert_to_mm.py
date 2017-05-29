import scipy.io as sio
from scipy.sparse import coo_matrix, csr_matrix
import numpy as np
import h5py

nusers = 458293
nitems = 17770

h5file = h5py.File('../mu/all.h5')

'''
probe_I = h5file['probe_user_list'][:]
probe_J = h5file['probe_item_list'][:]
probe_V = h5file['probe_rating_list'][:]

probe_V += 3.60951619727280626

sparseMat = csr_matrix((probe_V, (probe_I, probe_J)), shape=(nusers, nitems))

sio.mmwrite('probe_data.mm', sparseMat, field='integer')
'''

train_I = h5file['train_user_list'][:]
train_J = h5file['train_item_list'][:]
train_V = h5file['train_rating_list'][:]

train_V += 3.60951619727280626

sparseMatTrain = csr_matrix((train_V, (train_I, train_J)), shape=(nusers, nitems))

sio.mmwrite('../virtualbox_share/train_data.mm', sparseMatTrain, field='integer')

print ('Wrote train data')

qual_I = h5file['qual_user_list'][:]
qual_J = h5file['qual_item_list'][:]
qual_V = np.zeros(len(qual_I))

sparseMatQual = csr_matrix((qual_V, (qual_I, qual_J)), shape=(nusers, nitems))

sio.mmwrite('../virtualbox_share/qual_data.mm', sparseMatQual, field='integer')

print ('Wrote qual data')

del train_I
del train_J
del train_V
del sparseMatTrain

print ('deleted train matrices')

h5file_all = h5py.File('../mu/all_postprocess.h5')

all_I = h5file_all['all_user_list'][:]
all_J = h5file_all['all_item_list'][:]
all_V = h5file_all['all_rating_list'][:]

sparseMatAll = csr_matrix((all_V, (all_I, all_J)), shape=(nusers, nitems))

sio.mmwrite('../virtualbox_share/all_data.mm', sparseMatAll, field='integer')
print ('Wrote all data')
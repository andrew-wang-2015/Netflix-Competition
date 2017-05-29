import scipy.io as sio
from scipy.sparse import coo_matrix, csr_matrix
import numpy as np
import h5py
import envoy
import progressbar

nusers = 458293
nitems = 17770

#qual_mat = sio.mmread('../virtualbox_share/qual_data.mm.predict')

print('Read in Qual')
#qual_mat = qual_coo_mat.tocsr()

#qual_mat = qual_mat.reshape((nitems, nusers))
'''
qual_I = qual_mat.col
qual_J = qual_mat.row
qual_V = qual_mat.data
'''
#qual = np.empty((2749898, 3))

qual = []

f_qual = open('../virtualbox_share/qual_data.mm0.911.predict', "r")

for i, line in enumerate(f_qual.readlines()):
	if i > 2:
		line = line.split()
		line[0] = int(line[0])
		line[1] = int(line[1])
		line[2] = float(line[2])
		qual.append(line)
	#for i, word in line.sp

#qual_np = np.array(qual, dtype='float_')
qual_np = np.array(qual)
print (qual_np.shape)
print (qual_np[0])
print (qual_np[1])
print (qual_np[2])   
ind = np.lexsort((qual_np[:,0], qual_np[:,1]))
print (qual_np[ind[0]])
print (qual_np[ind[1]])
print (qual_np[ind[2]])
qual_np = qual_np[ind]
print (qual_np[0])
print (qual_np[1])
print (qual_np[2]) 

#qual = sorted(qual, key=lambda x: if x[1])
np.savetxt('qual_RBM_no_probe.txt', qual_np[:,2], fmt='%.3f')
#for 
print ('Saved Qual Predictions')

#print('User: ', all_I[0], 'Movie:', all_J[0],'Rating:', all_V[0])

dest_filename = '../mu/rbm_qual_no_probe.h5'

f = h5py.File(dest_filename, 'w')
#f.create_dataset('train_user_list', data = all_np[:,0])
#f.create_dataset('train_item_list', data = all_np[:,1])
#f.create_dataset('train_predictions', data = all_np)
#f.create_dataset('qual_user_list', data = qual_np[:,0])
#f.create_dataset('qual_item_list', data = qual_np[:,1])
f.create_dataset('qual_predictions', data = qual_np[:,2])
f.close()
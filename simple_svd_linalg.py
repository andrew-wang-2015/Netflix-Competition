# Aadith Moorthy

import h5py
import numpy as np
import scipy.sparse
from scipy.sparse.linalg import svds
from sklearn.decomposition import TruncatedSVD
import math
import os
import time
nusers = 458293
nitems = 17770

datafile = h5py.File('../mu/all.h5')




users = datafile['train_user_list']
items = datafile['train_item_list']

ratings = datafile['train_rating_list']

probe_users = datafile['probe_user_list']
probe_items = datafile['probe_item_list']

probe_ratings = datafile['probe_rating_list']



qual_users = datafile['qual_user_list']
qual_items = datafile['qual_item_list']



print 'loaded data'
R = scipy.sparse.coo_matrix((ratings, (items, users)), shape=(nitems, nusers), dtype='f')

print 'built sparse matrix'
k = 59
print k
R, s, vt = svds(R, k)

s_diag_matrix=np.diag(s)
print 'done svd and transform to latent var space'
R = np.dot(R, s_diag_matrix)
print 'done inner dot'





train_mse = 0.0
probe_mse = 0.0

batches = len(range(0, len(probe_ratings), 1000))
for i in range(0, len(probe_ratings), 1000):


    given = probe_ratings[i:min(i+1000, len(probe_ratings))]
    preding = np.einsum('ij,ji->i',R[probe_items[i:min(i+1000, len(probe_ratings))],:], vt[:,probe_users[i:min(i+1000, len(probe_ratings))]])

    inter_res = (given-preding)**2
    res = np.sum(inter_res)

    probe_mse += res

probe_mse /= len(probe_ratings)
print 'probe rmse', math.sqrt(probe_mse)
'''

for i in range(0, len(ratings), 1000):


    given = ratings[i:min(i+1000, len(ratings))]
    preding = np.einsum('ij,ji->i',R[items[i:min(i+1000, len(ratings))],:], vt[:,users[i:min(i+1000, len(ratings))]])

    inter_res = (given-preding)**2
    res = np.sum(inter_res)

    train_mse += res

train_mse /= len(ratings)
print 'train rmse', math.sqrt(train_mse)

pred = open('predictions.txt', 'w')
batches = len(range(0, len(qual_items), 1000))
for i in range(0, len(qual_items), 1000):


    preding = np.einsum('ij,ji->i',R[qual_items[i:min(i+1000, len(qual_items))],:], vt[:,qual_users[i:min(i+1000, len(qual_items))]])
    for pr in preding:
        pred.write('%.3f\n' % (pr+3+0.60951619727280626)) #convert back to 1-5 ratings systems
'''

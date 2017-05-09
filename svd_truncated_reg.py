# Aadith Moorthy

import h5py
import numpy as np
import scipy.sparse
#from scipy.sparse.linalg import svds
from sklearn.decomposition import TruncatedSVD
#from sklearn.linear_model import SGDRegressor
import math
import os
import time
nusers = 458293
nitems = 17770

datafile = h5py.File('../um/all.h5')




users = datafile['train_user_list']
items = datafile['train_item_list']

ratings = datafile['train_rating_list']

probe_users = datafile['probe_user_list']
probe_items = datafile['probe_item_list']

probe_ratings = datafile['probe_rating_list']



qual_users = datafile['qual_user_list']
qual_items = datafile['qual_item_list']



print 'loaded data'
R_m = scipy.sparse.csr_matrix((ratings, (items, users)), shape=(nitems, nusers))
R_u = scipy.sparse.csr_matrix((ratings, (users, items)), shape=(nusers, nitems))
print 'built sparse matrix'
k = 59
print k
svd = TruncatedSVD(k)
Rtr = svd.fit_transform(R_m)
print 'done transform'

probe_se = 0
for i in range(len(probe_ratings)):
    current_user = probe_users[i]
    user_movies_rated = R_u[current_user, :]
    row = np.concatenate((Rtr[user_movies_rated.indices, :],np.zeros((user_movies_rated.getnnz(),1))), axis = 1)

    ft, a, b, c = np.linalg.lstsq(row, user_movies_rated[0,user_movies_rated.indices].toarray().T)
    print np.dot(np.concatenate((Rtr[probe_items[i], :],[1])), ft)
    probe_se += (probe_ratings[i] - np.dot(np.concatenate((Rtr[probe_items[i], :],[1])), ft)[0])**2

    print probe_se/(i + 1), 'rre'


'''


### linear model to capture remaining effects
i = 0
lin_model = SGDRegressor()

rat_num = len(probe_ratings)
rat_num1000 = rat_num % 1000
for i in range(0, rat_num, 1000):

    given = probe_ratings[i:min(i+1000, rat_num)]
    try:
        preding = np.concatenate((np.einsum('ij,ji->i',R[probe_items[i:min(i+1000, rat_num)],:], vt[:,probe_users[i:min(i+1000, rat_num)]]).reshape(1000,1), np.ones((1000, 1))), axis = 1)
    except:
        preding = np.concatenate((np.einsum('ij,ji->i',R[probe_items[i:min(i+1000, rat_num)],:], vt[:,probe_users[i:min(i+1000, rat_num)]]).reshape(rat_num1000,1), np.ones((rat_num1000, 1))), axis = 1)
    lin_model.partial_fit(preding, given)

    if (i % 1000000) == 0:
        print lin_model.score(preding, given)


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
print 'pre-linear probe rmse', math.sqrt(probe_mse)

probe_mse = 0.0
rat_num = len(probe_ratings)
rat_num1000 = rat_num % 1000
for i in range(0, len(probe_ratings), 1000):


    given = probe_ratings[i:min(i+1000, len(probe_ratings))]
    try:
        preding = np.concatenate((np.einsum('ij,ji->i',R[probe_items[i:min(i+1000, rat_num)],:], vt[:,probe_users[i:min(i+1000, rat_num)]]).reshape(1000,1), np.ones((1000, 1))), axis = 1)
    except:
        preding = np.concatenate((np.einsum('ij,ji->i',R[probe_items[i:min(i+1000, rat_num)],:], vt[:,probe_users[i:min(i+1000, rat_num)]]).reshape(rat_num1000,1), np.ones((rat_num1000, 1))), axis = 1)
    preding = lin_model.predict(preding)
    inter_res = (given-preding)**2
    res = np.sum(inter_res)

    probe_mse += res

probe_mse /= len(probe_ratings)
print 'post-linear probe rmse', math.sqrt(probe_mse)




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
rat_num = len(qual_items)
rat_num1000 = rat_num % 1000
for i in range(0, len(qual_items), 1000):



    try:
        preding = np.concatenate((np.einsum('ij,ji->i',R[qual_items[i:min(i+1000, rat_num)],:], vt[:,qual_users[i:min(i+1000, rat_num)]]).reshape(1000,1), np.ones((1000, 1))), axis = 1)
    except:
        preding = np.concatenate((np.einsum('ij,ji->i',R[qual_items[i:min(i+1000, rat_num)],:], vt[:,qual_users[i:min(i+1000, rat_num)]]).reshape(rat_num1000,1), np.ones((rat_num1000, 1))), axis = 1)
    preding = lin_model.predict(preding)
    for pr in preding:
        pred.write('%.3f\n' % max(min((pr+3+0.60951619727280626),5),1)) #convert back to 1-5 ratings systems
'''

# 0s test - what happens if we just submit zeros for current data
# Aadith Moorthy
import h5py
import numpy as np
import math
h5filename = "../mu/svd_reduced_ratings"

h5file = h5py.File(h5filename)


ratings = h5file['train_rating_list']

probe_ratings = h5file['probe_rating_list']

all_ratings = probe_ratings#np.concatenate((ratings, probe_ratings))
print 'data mean', np.mean(all_ratings)

num = len(all_ratings)
batch_size = 10000
se = 0
for i in range(0, num, batch_size):

    pred = 0
    #print u_mean, m_mean, pred
    given = all_ratings[i:min(i+batch_size, num)]

    se += np.sum(np.power((given-pred),2))

    #print math.sqrt(probe_se/(i+batch_size))



print 'Submitting just zeros, you get a total RMSE of:', math.sqrt(se/num)

# preliminary booster, getting resid of ratings for more trainin

from keras.models import load_model
model = load_model('svd_keras0.2.mdl')
import h5py
import numpy as np
import gnumpy as gpu
datafile = h5py.File('../mu/all_baselined_bellkor96.h5')

users = datafile['train_user_list']
items = datafile['train_item_list']

ratings = datafile['train_rating_list'][:]

probe_users = datafile['probe_user_list']
probe_items = datafile['probe_item_list']

probe_ratings = datafile['probe_rating_list'][:]

pred_ratings = model.predict([users, items], batch_size=100000)

new_ratings = np.zeros(len(ratings))
print ratings.shape, pred_ratings.shape

for i in range(0, len(ratings), 10000):
    new_ratings[i:min(i +10000,  len(ratings) )] =  gpu.garray(ratings[i:min(i +10000,  len(ratings) )]) - gpu.garray(pred_ratings[i:min(i +10000,  len(ratings) )].T)


pred_probe_ratings = model.predict([probe_users, probe_items], batch_size=100000)


new_probe_ratings = np.zeros(len(probe_ratings))

for i in range(0, len(probe_ratings), 10000):
    new_probe_ratings[i:min(i +10000,  len(probe_ratings) )] =  gpu.garray(probe_ratings[i:min(i +10000,  len(probe_ratings) )]) - gpu.garray(pred_probe_ratings[i:min(i +10000,  len(probe_ratings) )].T)

resids = h5py.File('svd_reduced_ratings')
resids.create_dataset('train_rating_list', data = new_ratings)
resids.create_dataset('probe_rating_list', data = new_probe_ratings)
resids.close()
#np.savetxt('new_ratings.txt', new_ratings)
#np.savetxt('new_probe_ratings.txt', new_probe_ratings)

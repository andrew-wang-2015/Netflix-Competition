import numpy as np
import h5py
#import gnumpy as gpu
from scipy.sparse import coo_matrix, csr_matrix
import math
import progressbar

h5file = h5py.File('../mu/all.h5')
#datafile_alt = h5py.File('../mu/all_baselined.h5')
h5file_all = h5py.File('../mu/all_postprocess.h5')


nusers = 458293
nitems = 17770
'''
I = h5file['train_user_list']
J = h5file['train_item_list']
V = h5file['train_rating_list']

probe_I = h5file['probe_user_list']
probe_J = h5file['probe_item_list']
probe_V = h5file['probe_rating_list']
'''
all_I = h5file_all['all_user_list']
all_J = h5file_all['all_item_list']
all_V = h5file_all['all_rating_list']


qual_I = h5file['qual_user_list']
qual_J = h5file['qual_item_list']


#movie_avg = datafile_alt['train_item_avg'][:]
print ('Setting up Dictionaries')
bar = progressbar.ProgressBar(maxval=len(all_I), widgets=["Setting up Dictionaries: ", progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), \
    ' ', progressbar.ETA()]).start()
users_rated = np.zeros(nusers)

for user in range(nusers):
	users_rated[user] = 0

for i, user in np.ndenumerate(all_I):
	if (i[0] % 1000) == 0:
		bar.update(i[0] % bar.max_value)
	users_rated[user] = 1
bar.finish()
'''
for user in probe_I:
	users_rated[user] = 1
'''
print ('Done setting up dictionaries')

u_pred = np.loadtxt('../predictions.txt')

print ('Loaded in predictions')
'''

bar = progressbar.ProgressBar(maxval=len(all_I), widgets=["Setting up Dictionaries: ", progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), \
    ' ', progressbar.ETA()]).start()

all_I = np.zeros(len(probe_I) + len(I))
all_J = np.zeros(len(probe_J) + len(J))
all_V = np.zeros(len(probe_V) + len(V))

length_I = len(I)

for i in range(len(probe_I) + len(I)):
	if (i % 1000) == 0:
		bar.update(i % bar.max_value)
	if i < length_I:
		all_I[i] = I[i]
		all_J[i] = J[i]
		all_V[i] = V[i]
	else:
		all_I[i] = probe_I[i - length_I]
		all_J[i] = probe_J[i - length_I]
		all_V[i] = probe_V[i - length_I]
bar.finish()
print('Created all data array')
'''
Pred_R_m = csr_matrix((u_pred, (qual_J, qual_I)), shape=(nitems, nusers))
print ('Made Prediction CSR matrix')
R_m = csr_matrix((all_V, (all_J, all_I)), shape=(nitems, nusers))
print ('Made Train CSR matrix')


# Predicted Movie Averahe
pred_movie_avg = np.zeros(nitems)
batch_size = 10000
for i in range(0, nitems, batch_size):

    movie_current = Pred_R_m[i:min(i+batch_size, nitems), :]
    batch_avg = ((movie_current.sum(axis =1).flatten())/movie_current.getnnz(axis =1))
    pred_movie_avg[i:min(i+batch_size, nitems)] = batch_avg

print ('compiled prediction averages')
movie_avg = np.zeros(nitems)
batch_size = 10000
for i in range(0, nitems, batch_size):

    movie_current = R_m[i:min(i+batch_size, nitems), :]
    batch_avg = ((movie_current.sum(axis =1).flatten())/movie_current.getnnz(axis =1))
    movie_avg[i:min(i+batch_size, nitems)] = batch_avg

print ('compiled train averages')
print ("MADE IT!")

numQual = len(qual_I)

new_pred = np.zeros(numQual)
bar = progressbar.ProgressBar(maxval=numQual, widgets=["Changing Predictions: ", progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), \
    ' ', progressbar.ETA()]).start()

#fin_pred = open('final_predictions.txt', 'w')
for i, pred in np.ndenumerate(u_pred):
	if (i[0] % 1000) == 0:
		bar.update(i[0] % bar.max_value)
	curr_pred = pred
	user = qual_I[i]
	movie = qual_J[i]
	#print(user)
	if users_rated[user] == 0:
		print ("No ratings for this user: ", user)
		curr_pred = movie_avg[movie]
	if curr_pred > 5:
		curr_pred = 5
	if curr_pred < 1:
		curr_pred = 1
	curr_pred += movie_avg[movie] - pred_movie_avg[movie]
	decimal = curr_pred - int(curr_pred)
	if decimal <= 0.1:
		curr_pred = int(curr_pred)
	if decimal >= 0.9:
		curr_pred = int(curr_pred) + 1
	new_pred[i] = curr_pred
	#pred.write('%.3f\n' % curr_pred)
bar.finish()

np.savetxt('final_predictions.txt', new_pred, fmt='%.3f')
'''
fin_pred = open('final_predictions.txt', 'w')

bar = progressbar.ProgressBar(maxval=numQual, widgets=["Writing Predictions: ", progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), \
    ' ', progressbar.ETA()]).start()

for i, pr in np.ndenumerate(new_pred):
	if (i[0] % 1000) == 0:
		bar.update(i[0] % bar.max_value)
	pred.write('%.3f\n' % float(pr))

bar.finish()
'''
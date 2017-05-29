import numpy as np
import h5py
#import gnumpy as gpu
from scipy.sparse import coo_matrix, csr_matrix
import math
import progressbar

h5file = h5py.File('../mu/all.h5')
#datafile_alt = h5py.File('../mu/all_baselined.h5')
h5file_all = h5py.File('../mu/all_postprocess.h5')

h5file_new = h5py.File('../mu/svd_resids903.h5')
for key in h5file_new.keys():
	print (key)
	print (h5file_new[key].shape)
	print (h5file_new[key][0])
	print (h5file_new[key][0])

u_pred_probe = h5file_new['probe_predictions'][:]
#probe_resids = h5file_new['probe_rating_list'][:]

h5file_rbm_yes_probe = h5py.File('../mu/rbm_all.h5')

rbm_all_pred_yes_probe = h5file_rbm_yes_probe['train_predictions'][:]

h5file_rbm_on_train = h5py.File('../mu/rbm_all_on_train.h5')

rbm_all_pred_on_train = h5file_rbm_on_train['train_predictions'][:]



nusers = 458293
nitems = 17770


I = h5file['train_user_list'][:]
J = h5file['train_item_list'][:]
V = h5file['train_rating_list'][:]

probe_I = h5file['probe_user_list'][:]
probe_J = h5file['probe_item_list'][:]
#probe_V = h5file['probe_rating_list'][:]

V += 3.60951619727280626
#probe_V += 3.60951619727280626
'''
print (V[0])
print ('rating - residual: ', probe_V[0] - probe_resids[0], ', new rating: ', probe_predictions[0])
print ('rating - residual: ', probe_V[1] - probe_resids[1], ', new rating: ', probe_predictions[1])
'''
'''
all_I = h5file_all['all_user_list'][:]
all_J = h5file_all['all_item_list'][:]
all_V = h5file_all['all_rating_list'][:]
'''
train_predictions = V - h5file_new['train_rating_list'][:]

del V

print (train_predictions[0])
train_predictions = train_predictions * 0.98 + 0.01 * rbm_all_pred_yes_probe + 0.01 * rbm_all_pred_on_train
print (train_predictions[0])
#qual_I = h5file['qual_user_list'][:]
#qual_J = h5file['qual_item_list'][:]

print ('Done setting up dictionaries')

print ('Loaded in predictions')

# Combining Probe and Train Predictions
'''
total_len = len(probe_I) + len(I)
bar = progressbar.ProgressBar(maxval=total_len, widgets=["Getting probe and train predictions: ", progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), \
    ' ', progressbar.ETA()]).start()

pred_I = np.empty(total_len)
pred_J = np.empty(total_len)
pred_V = np.empty(total_len)

length_I = len(I)

for i in range(total_len):
	if (i % 1000) == 0:
		bar.update(i % bar.max_value)
	if i < length_I:
		pred_I[i] = I[i]
		pred_J[i] = J[i]
		pred_V[i] = train_predictions[i]
	else:
		pred_I[i] = probe_I[i - length_I]
		pred_J[i] = probe_J[i - length_I]
		pred_V[i] = u_pred_probe[i - length_I]
bar.finish()
'''
print('Created pred data array')

del train_predictions
del I
del J
del probe_I
del probe_J
del u_pred_probe

print ('Done with deletes')
#Pred_R_m = csr_matrix((u_pred, (qual_J, qual_I)), shape=(nitems, nusers))
'''
Pred_R_m = csr_matrix((pred_V, (pred_J, pred_I)), shape=(nitems, nusers))
print ('Made Prediction CSR matrix')
R_m = csr_matrix((all_V, (all_J, all_I)), shape=(nitems, nusers))
print ('Made Train CSR matrix')
'''

pred_I = np.empty(total_len)
pred_J = np.empty(total_len)
pred_V = np.empty(total_len)

Pred_R_u = csr_matrix((pred_V, (pred_I, pred_J)), shape=(nusers, nitems))
print ('Made Prediction CSR matrix')

#del pred_I
#del pred_J
#del pred_V
pred_user_avg = np.zeros(nusers)
batch_size = 10000
for i in range(0, nusers, batch_size):
    users_current = Pred_R_u[i:min(i+batch_size, nusers), :]
    batch_avg = ((users_current.sum(axis =1).flatten())/users_current.getnnz(axis =1))
    pred_user_avg[i:min(i+batch_size, nusers)] = batch_avg

del pred_I
del pred_J
del pred_V
del Pred_R_u
print ('Deleted references to prediction CSR')

all_I = h5file_all['all_user_list'][:]
all_J = h5file_all['all_item_list'][:]
all_V = h5file_all['all_rating_list'][:]

R_u = csr_matrix((all_V,(all_I,all_J)), shape=(nusers, nitems))
print ('Made Train CSR matrix')

user_avg = np.zeros(nusers)
batch_size = 10000
for i in range(0, nusers, batch_size):
    users_current = R_u[i:min(i+batch_size, nusers), :]
    batch_avg = ((users_current.sum(axis =1).flatten())/users_current.getnnz(axis =1))
    user_avg[i:min(i+batch_size, nusers)] = batch_avg
'''
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
'''
qual_I = h5file['qual_user_list'][:]
qual_J = h5file['qual_item_list'][:]

u_pred_qual = np.loadtxt('../mu/svd_preds903.txt')
#u_pred_probe = np.loadtxt('probe_predictions.txt')

print ('Loaded in predictions')
numQual = len(qual_I)

new_pred = np.zeros(numQual)
bar = progressbar.ProgressBar(maxval=numQual, widgets=["Changing Predictions: ", progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), \
    ' ', progressbar.ETA()]).start()

#fin_pred = open('final_predictions.txt', 'w')
for i, pred in np.ndenumerate(u_pred_qual):
	if (i[0] % 1000) == 0:
		bar.update(i[0] % bar.max_value)
	curr_pred = pred
	user = qual_I[i]
	movie = qual_J[i]
	#print(user)
	'''
	if users_rated[user] == 0:
		print ("No ratings for this user: ", user)
		curr_pred = movie_avg[movie]
	'''
	if curr_pred > 5:
		curr_pred = 5
	if curr_pred < 1:
		curr_pred = 1
	#curr_pred += movie_avg[movie] - pred_movie_avg[movie]
	curr_pred += user_avg[user] - pred_user_avg[user]
	'''
	decimal = curr_pred - int(curr_pred)
	if decimal <= 0.1:
		curr_pred = int(curr_pred)
	if decimal >= 0.9:
		curr_pred = int(curr_pred) + 1
	'''
	new_pred[i] = curr_pred
	#pred.write('%.3f\n' % curr_pred)
bar.finish()

np.savetxt('postprocessed_predictions_qual_user.txt', new_pred, fmt='%.3f')


dest_filename = '../mu/qual_postprocess_user.h5'

f = h5py.File(dest_filename, 'w')
f.create_dataset('qual_predictions', data = new_pred)
f.close()
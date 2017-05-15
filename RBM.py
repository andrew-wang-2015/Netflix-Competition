from __future__ import print_function
#import get_data_progbar2
import scipy.sparse
'''
from sklearn.neural_network import BernoulliRBM
from sklearn.preprocessing import normalize
from sklearn import linear_model
'''
import numpy as np
import envoy
import progressbar
import h5py
import math
import random
import tensorflow as tf
#import theano

import sys
#sys.path.append('C:/Users/dck25/Documents/Caltech/CS156b/rbm.py/rbm')
#sys.path.append('C:/Users/dck25/Documents/Caltech/CS156b/')
sys.path.append('/home/ec2-user/CS156b/')
#sys.path.append('/home/ec2-user/CS156b/rbm.py/rbm')

#import rbm
import tensor_rbm

h5filename = "../mu/all.h5"

h5_V_filename = "../mu/user_V.h5"

h5file = h5py.File(h5filename)

h5file_V = h5py.File(h5_V_filename)

I = h5file['train_user_list']
J = h5file['train_item_list']
V = h5file['train_rating_list']

probe_I = h5file['probe_user_list']
probe_J = h5file['probe_item_list']
probe_V = h5file['probe_rating_list']


qual_users = h5file['qual_user_list']
qual_items = h5file['qual_item_list']

user_V_I = h5file_V['train_user_rating_list']
user_V_J = h5file_V['train_item_list']
user_V_V = h5file_V['train_binary_list']
#RBM_rows = h5file['train_RBM_rows_list']

numUsers = 458293
numMovies = 17770
dataPoints = len(I)
batch_size = 1000 # {1, 7, 31, 217, 452957, 3170699, 14041667, 98291669}
#num_epochs = 10
numFactors = 20

hidden_probs_final = np.zeros((numUsers * 5, numFactors))

user_sparse_matrix = scipy.sparse.coo_matrix(
        (V, (I, J)), shape=(numUsers, numMovies))

probe_sparse_matrix = scipy.sparse.coo_matrix(
        (probe_V, (probe_I, probe_J)), shape=(numUsers, numMovies))

user_mat = scipy.sparse.coo_matrix(
        (user_V_V, (user_V_I, user_V_J)), shape=(numUsers * 5, numMovies))

def sigmoid_approx(x):
  return x / (1 + abs(x))
# Convert a specific user's ratings to a binary K x M matrix, where K = number of different ratings 
# and M = number of movies that this user rated.
def convert_to_V(user_sm):
  #user_ratings = user_sm.getrow(user_num)
  num_ratings = user_sm.getnnz()
  #movie_ids = np.zeros(num_ratings)
  orig_data = user_sm.data
  orig_row = user_sm.row
  orig_col = user_sm.col
  fin_row = np.zeros(num_ratings)
  #col = np.zeros(num_ratings)
  fin_col = orig_col
  fin_data = np.ones(num_ratings)
  #count = 0
  bar = progressbar.ProgressBar(maxval=num_ratings, widgets=["Converting to V: ", progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), \
    ' ', progressbar.ETA()]).start()
  for i, rating in np.ndenumerate(orig_data):
    if (i[0] % 1000) == 0:
      bar.update(i[0] % bar.max_value)
    fin_row[i[0]] = (orig_row[i[0]]) * 5 + (rating - 1)
    #fin_col[count] = count
    # i is tuple of index in original matrix
    #movie_ids[count] = i[1]
    #count += 1

  bar.finish()
  f = h5py.File('../mu/user_V.h5', 'w')
  f.create_dataset('train_user_rating_list', data = fin_row)
  f.create_dataset('train_item_list', data = fin_col)
  f.create_dataset('train_binary_list', data = fin_data)
  f.close()
  return scipy.sparse.coo_matrix((fin_data,(fin_row,fin_col)), shape=(user_sm.shape[0] * 5, user_sm.shape[1]))

'''
def predict_new_rating(model, weights, visible_bias, user_id, movie_id):
  # Produce vector P_hat[j] of p(h_j = 1 | V) for each factor j
  #print ('User ID: ', user_id, 'Movie ID: ', movie_id)
  numerators = np.zeros(5)
  for rating in range(1, 6):
    P_hat = model.get_hidden_probs(user_mat.getrow(user_id * 5 + (rating - 1)).todense())
    #print(P_hat.shape)
    #print(P_hat)
    #P_hat = np.asarray(P_hat);
    #P_hat = hidden_states[user_id * 5 + (rating - 1)]
    numerators[rating - 1] = visible_bias[movie_id]
    for j in range(len(P_hat)):
      numerators[rating - 1] += P_hat[0][j] * weights[j][movie_id]
    numerators[rating - 1] = np.exp(numerators[rating - 1])

  denomin = np.sum(numerators)
  final_rate = 0
  for i in range(5):
    #print (numerators[i])
    final_rate += numerators[i] * (i + 1) / denomin
  return final_rate
'''
def populate_hidden_probs(model, weights, hidden_bias):
  hidden_bar = progressbar.ProgressBar(maxval=numUsers, widgets=["Calculating Hidden Probs: ", progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), \
        ' ', progressbar.ETA()]).start()
  all_probs = []
  with tf.Session() as sess:
      for i in range(numUsers):
          if (i % 1000) == 0:
            hidden_bar.update(i % hidden_bar.max_value)
          hprobs = tf.nn.sigmoid(tf.matmul(tf.cast(user_mat[i * 5: 5 * (i + 1)].todense(), tf.float32), \
       weights + hidden_bias))
          all_probs.append(hprobs)
          #user_rows = hprobs.eval()
          #hidden_probs_final[i * 5: (i + 1) * 5] = user_rows
      all_probs = sess.run(all_probs)
      hidden_bar.finish()
      for j in range(len(all_probs)):
          hidden_probs_final[j * 5: (j + 1) * 5] = all_probs[j]

def do_hidden_probs_1(visible, weights, hidden_bias):
    with tf.Session() as sess:
        coo = visible[:(numUsers * 3)].tocoo()
        indices = np.mat([coo.row, coo.col]).transpose()
        visible = tf.SparseTensor(indices, coo.data, coo.shape)
        #visible = convert_sparse_matrix_to_sparse_tensor(visible)
        hprobs = tf.nn.sigmoid(tf.sparse_tensor_dense_matmul(tf.cast(visible, tf.float32), weights) + hidden_bias)
        hidden_probs_final[:(numUsers * 3)] = hprobs.eval()
        #return hprobs.eval()

def do_hidden_probs_2(visible, weights, hidden_bias):
    tf.reset_default_graph()
    with tf.Session() as sess:
        coo = visible[(numUsers * 3):(numUsers * 5)].tocoo()
        indices = np.mat([coo.row, coo.col]).transpose()
        visible = tf.SparseTensor(indices, coo.data, coo.shape)
        #visible = convert_sparse_matrix_to_sparse_tensor(visible)
        hprobs = tf.nn.sigmoid(tf.sparse_tensor_dense_matmul(tf.cast(visible, tf.float32), weights) + hidden_bias)
        hidden_probs_final[(numUsers * 3): (numUsers * 5)] = hprobs.eval()
        #return hprobs.eval()


      #hidden_bar.finish()
def predict_new_rating(model, weights, hidden_bias, visible_bias, user_id, movie_id):
  # Produce vector P_hat[j] of p(h_j = 1 | V) for each factor j
  #print ('User ID: ', user_id, 'Movie ID: ', movie_id)
  #with tf.Session() as sess:
    #hprobs = tf.nn.sigmoid(tf.matmul(tf.cast(visible, tf.float32), self.W) + self.bh_)
    #P_hat = hprobs.eval()
    numerators = np.zeros(5)
    for rating in range(1, 6):
      '''
      hprobs = tf.nn.sigmoid(tf.matmul(tf.cast(user_mat.getrow(user_id * 5 + (rating - 1)).todense(), tf.float32), \
       weights + hidden_bias))
      '''
      #P_hat = hprobs.eval()
      #P_hat = model.get_hidden_probs(user_mat.getrow(user_id * 5 + (rating - 1)).todense())
      #print(P_hat.shape)
      #print(P_hat)
      #P_hat = np.asarray(P_hat);
      #P_hat = hidden_states[user_id * 5 + (rating - 1)]
      P_hat = hidden_probs_final[user_id * 5 + (rating - 1)]
      numerators[rating - 1] = visible_bias[movie_id]
      for j in range(len(P_hat)):
        numerators[rating - 1] += P_hat[j] * weights[movie_id][j]
      numerators[rating - 1] = np.exp(numerators[rating - 1])

    denomin = np.sum(numerators)
    final_rate = 0
    for i in range(5):
      #print (numerators[i])
      final_rate += (numerators[i] * (i + 1)) / denomin
    return final_rate



# TRain RBM
if __name__ == '__main__':

  # Data is made continuous for continuous RBM
  #user_sparse_matrix = get_data_progbar2.get_train_data_user('../mu/train.dta')
  user_sparse_matrix.data += 3.7
  user_sparse_matrix = user_sparse_matrix.floor()

  probe_sparse_matrix.data += 3.7
  probe_sparse_matrix = probe_sparse_matrix.floor()
  
  print ('Converted all ratings: ')
  # user_mat = convert_to_V(user_sparse_matrix)
  # Load RBM binary matrix
  '''
  fin_data = np.ones(user_sparse_matrix.getnnz())
  fin_row = RBM_rows
  user_mat = scipy.sparse.coo_matrix((fin_data,(fin_row,user_sparse_matrix.col)), shape=(user_sparse_matrix.shape[0] * 5, user_sparse_matrix.shape[1]))
  '''
  #print (user_mat.getnnz())
  user_num_nnz = user_mat.getnnz()
  probe_num_nnz = len(probe_I)
  #probe_mat = convert_to_V(probe_sparse_matrix)
  #user_mat = convert_to_V(user_sparse_matrix)
  '''model_t = t_RBM.RBM(17770, 100, visible_unit_type='bin', main_dir='rbm', model_name='rbm_model',
                 gibbs_sampling_steps=1, learning_rate=0.01, batch_size=1000, num_epochs=10, stddev=0.1, verbose=1)

  model_t.fit(user_mat)
  '''


  #probe_mat = probe_mat.tocsr()
  user_mat = user_mat.tocsr()
  #user_mat = user_mat[:150]
  #probe_mat = probe_mat[:150]
  #probe_mat = probe_mat.tocoo()
  #print (probe_mat.nnz)
  print('Matrix form changed \n')

  '''
  model_test = BernoulliRBM(n_components = 100, verbose=1)
  model_test.fit(probe_mat)
  '''

  model = tensor_rbm.RBM(numMovies, numFactors, verbose=1)
  #model = model.load_model((numMovies, 20), 1, 'models/rbm')
  print ('Getting Model Ready: ')
  model.fit(user_mat)
  #print(model.components_.shape)
  model_comps = model.get_model_parameters()
  model_weights = model_comps['W']
  model_vis_bias = model_comps['bv_']
  model_hid_bias = model_comps['bh_']
  print('Weight Matrix: \n', model_weights)
  print('Visible Biases: \n', model_vis_bias)
  print('Hidden Biases: \n', model_hid_bias)

  #hidden_probs = model.get_hidden_probs(user_mat)
  #hidden_probs = model_comps['hidden_probs']
  #hidden_probs = model.transform(user_mat)
  #hidden_probs = user_mat.multiply(model_weights)
  #print('Hidden Units: \n', hidden_probs)

  '''
  hidden_probs = model_test.transform(probe_mat)
  model_weights = model_test.components_
  model_vis_bias = model_test.intercept_visible_
  '''
  do_hidden_probs_1(user_mat, model_weights, model_hid_bias)
  do_hidden_probs_2(user_mat, model_weights, model_hid_bias)

  # Calculate train RMSE
  #with tf.Session() as sess:
  
  train_sq_error = 0
  predict_bar = progressbar.ProgressBar(maxval=probe_num_nnz, widgets=["Predicting Train: ", progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), \
    ' ', progressbar.ETA()]).start()
  for i in range(probe_num_nnz):
    if (i % 1000) == 0:
      predict_bar.update(i % predict_bar.max_value)
    new_rating = predict_new_rating(model, model_weights, model_hid_bias, model_vis_bias, user_sparse_matrix.row[i], user_sparse_matrix.col[i])
    old_rating = user_sparse_matrix.data[i]
    #print ('New Rating: ', new_rating)
    #print ('Actual Rating: ', old_rating)
    train_sq_error += (new_rating - old_rating) ** 2
  predict_bar.finish()
  train_rmse = np.sqrt(train_sq_error / probe_num_nnz)
  print ('Train RMSE: ' , train_rmse)
  
  # Calculate probe RMSE
  probe_sq_error = 0
  predict_probe_bar = progressbar.ProgressBar(maxval=probe_num_nnz, widgets=["Predicting Probe: ", progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), \
    ' ', progressbar.ETA()]).start()
  for i in range(probe_num_nnz): #probe_num_nnz
    if (i % 1000) == 0:
        predict_probe_bar.update(i % predict_probe_bar.max_value)
    new_rating = predict_new_rating(model, model_weights, model_hid_bias, model_vis_bias, probe_sparse_matrix.row[i], probe_sparse_matrix.col[i])
    old_rating = probe_sparse_matrix.data[i]
    #print ('New Rating: ', new_rating)
    #print ('Actual Rating: ', old_rating)
    probe_sq_error += (new_rating - old_rating) ** 2
  predict_probe_bar.finish()
  probe_rmse = np.sqrt(probe_sq_error / probe_num_nnz)
  print ('Probe RMSE: ' , probe_rmse)

  predict_qual_bar = progressbar.ProgressBar(maxval=len(qual_users), widgets=["Predicting Qual: ", progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), \
    ' ', progressbar.ETA()]).start()
  pred = open('predictions.txt', 'w')
  for i in range(len(qual_users)):
    if (i % 1000) == 0:
        predict_qual_bar.update(i % predict_qual_bar.max_value)
    new_rating = predict_new_rating(model, model_weights, model_hid_bias, model_vis_bias, qual_users[i], qual_items[i])
    pred.write('%.3f\n' % (new_rating)) # Write new prediction to file
  predict_qual_bar.finish()

  #r.train(user_0_mat, max_epochs = 5000)

  #r = RBM(num_visible = 6, num_hidden = 2)
  '''training_data = np.array([[1,1,1,0,0,0],[1,0,1,0,0,0],[1,1,1,0,0,0],[0,0,1,1,1,0], [0,0,1,1,0,0],[0,0,1,1,1,0]])
  r.train(training_data, max_epochs = 5000)
  print(r.weights)
  user = np.array([[0,0,0,1,1,0]])
  print(r.run_visible(user))'''

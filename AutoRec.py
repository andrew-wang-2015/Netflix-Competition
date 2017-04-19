# coding=utf8
# Implementation of AutoRec in tf. See http://users.cecs.anu.edu.au/~u5098633/papers/www15.pdf
# Aadith Moorthy

import tensorflow as tf

from get_data import import_data_movie
import matplotlib.pyplot as plt
import math
import numpy as np
import envoy
import random
import scipy.sparse
import progressbar

nusers = 458293
nitems = 17770

batch_size = 20 # divisors of nitems: {1, 2, 5, 10, 1777, 3554, 8885, 17770}
num_epochs = 5

# hyperparameters
latent_vars = 50
learning_rate = 0.1

datafile = h5py.File('../mu/all.h5')

users = datafile['train_user_list']
items = datafile['train_item_list']

ratings = datafile['train_rating_list']

probe_users = datafile['probe_user_list']
probe_items = datafile['probe_item_list']

probe_ratings = datafile['probe_rating_list']

# convert to 2d objects for convenience
start = 0
users_aux = []
ratings_aux = []
for item in range(nitems):
    end = np.searchsorted(items, item + 1)
    users_aux.append(users[start:end])

    ratings_aux.append(ratings[start:end])

users = users_aux
ratings = ratings_aux

start = 0
probe_users_aux = []
ratings_aux = []
for item in range(nitems):
    end = np.searchsorted(probe_items, item + 1)
    probe_users_aux.append(users[start:end])

    probe_ratings_aux.append(ratings[start:end])

probe_users = probe_users_aux
probe_ratings = probe_ratings_aux

print "loaded all data"
# Set seeds for random numbers to ensure reproducibility of results
np.random.seed(11261996)
tf.set_random_seed(11261996)
random.seed(11261996)


# We create placeholders to correctly implementing the 'masking' feature
# When training, only want to update the weights corresponding to observed param
# This is the hardest part
train_data = tf.placeholder(tf.float32, [None, nusers])
train_observations = tf.placeholder(tf.float32, [None, nusers]) # the 'mask'

# layers
# scale to allow variation:
scale = math.sqrt(6.0 / (nusers + latent_vars)) # not 100% sure why this

# following notation from paper
# layer 1
V = tf.Variable(tf.random_uniform([nusers, latent_vars], -scale, scale))
u = tf.Variable(tf.random_uniform([latent_vars], -scale, scale))
g = tf.nn.softmax(tf.matmul(train_data, V) + u)
#activation here can be changed as needed

# layer 2
W = tf.Variable(tf.random_uniform([latent_vars, nusers], -scale, scale))
b = tf.Variable(tf.random_uniform([nusers], -scale, scale))
f = tf.matmul(g, W) + b #no activation because ratings are -2,-1,0,1,2

test_data = tf.placeholder(tf.float32, [None, nusers])
test_observations = tf.placeholder(tf.float32, [None, nusers])
squared_error_tf = tf.reduce_sum(tf.square((f - test_data)*test_observations))

tf.add_to_collection('predictor', f)
tf.add_to_collection('squared_error_calc', squared_error_tf)
# loss is masked to only include observed
masked_loss = tf.reduce_mean(tf.reduce_sum(tf.square((f - train_data)*train_observations), 1, keep_dims=True))
trainStep = tf.train.RMSPropOptimizer(learning_rate).minimize(masked_loss)

# start tf session and training
saver = tf.train.Saver()
autoRecSession = tf.InteractiveSession()
autoRecSession.run(tf.initialize_all_variables())

# go through epochs
# randomly iterating through items
# it is found in the paper that items are better to put as data points instead of users
items_list = range(nitems)

print "Starting training epochs"
probe_rmse_lst = []
train_rmse_lst = []
datapr_num = datapr.getnnz()
datatr_num = datatr.getnnz()
try:
    for epoch in range(num_epochs):
        print "Epoch %d of %d" % (epoch +1, num_epochs)
        random.shuffle(items_list)
        batch_num = int(math.ceil(nitems / float(batch_size)))

        bar = progressbar.ProgressBar(maxval=batch_num, widgets=["Training: ",
                                                                 progressbar.Bar(
                                                                     '=', '[', ']'),
                                                                 ' ', progressbar.Percentage(),

                                                                 ' ', progressbar.ETA()]).start()
        for batch in range(batch_num):

            if (batch % (int(0.01*batch_num))) == 0:
                bar.update(batch % bar.maxval)

            start_idx = batch * batch_size
            end_idx = min(start_idx + batch_size, nitems)
            #print batch, start_idx, end_idx
            batch_data = []
            batch_observations = []
            for item_idx in range(start_idx, end_idx):
                item = items_list[item_idx]
                item_features = np.zeros((nusers))
                item_features[users[item]] = ratings[item]
                batch_data.append(item_features)
                item_observation = np.zeros(nusers)
                item_observation[users[item]] = 1
                batch_observations.append(item_observation)

            batch_data = np.array(batch_data)

            batch_observations = np.array(batch_observations)

            trainStep.run(feed_dict={train_data:batch_data, train_observations:batch_observations})
        bar.finish()


        # to get proper test data, read as needed
        # do 1 batch at a time to save memory
        probe_squared_error = 0.0
        train_squared_error = 0.0

        bar = progressbar.ProgressBar(maxval=batch_num, widgets=["Testing: ",
                                                                 progressbar.Bar(
                                                                     '=', '[', ']'),
                                                                 ' ', progressbar.Percentage(),

                                                                 ' ', progressbar.ETA()]).start()

        for batch in range(batch_num):

            if (batch % (int(0.01*batch_num))) == 0:
                bar.update(batch % bar.maxval)

            start_idx = batch * batch_size
            end_idx = min(start_idx + batch_size, nitems)
            #print batch, start_idx, end_idx
            datapr_all = []
            datatr_all = []
            datatr_observation = []
            datapr_observation = []
            for item_idx in range(start_idx, end_idx):

                datapr_row = datapr.getrow(item_idx)
                datapr_all.append(datapr_row.toarray().flatten())
                datatr_row = datatr.getrow(item_idx)
                datatr_all.append(datatr_row.toarray().flatten())
                datapr_observation_row = np.zeros((nusers))
                datatr_observation_row = np.zeros((nusers))
                datapr_observation_row[datapr_row.indices] = 1
                datapr_observation.append(datapr_observation_row)
                datatr_observation_row[datatr_row.indices] = 1
                datatr_observation.append(datatr_observation_row)
            datapr_all = np.array(datapr_all)
            datatr_all = np.array(datatr_all)
            datatr_observation = np.array(datatr_observation)
            datapr_observation = np.array(datapr_observation)
            probe_squared_error += squared_error_tf.eval(feed_dict={train_data:datatr_all, test_data:datapr_all, test_observations:datapr_observation})
            train_squared_error += squared_error_tf.eval(feed_dict={train_data:datatr_all, test_data:datatr_all, test_observations:datatr_observation})

        bar.finish()
        probe_rmse_lst.append(math.sqrt(probe_squared_error/datapr_num))
        train_rmse_lst.append(math.sqrt(train_squared_error/datatr_num))
        print "Finished epoch %d of %d with train rmse of %.4f and probe rmse of %.4f" % (epoch+1, num_epochs, train_rmse_lst[-1],probe_rmse_lst[-1])
except KeyboardInterrupt:
    print 'saving progress and closing'

saver.save(autoRecSession, 'AutoRec.mdl')

f = plt.figure(1)
# take negative to get lower bound
plt.plot(range(1,len(probe_rmse_lst)+1), probe_rmse_lst, label='Probe_rmse')

plt.xlabel('Epoch')
plt.legend()
plt.ylim((.5,1))
plt.ylabel('I-AutoRec RMSE')

f.savefig('i_autorec_rmse.pdf')

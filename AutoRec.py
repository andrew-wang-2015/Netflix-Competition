# coding=utf8
# Implementation of AutoRec in tf. See http://users.cecs.anu.edu.au/~u5098633/papers/www15.pdf
# Aadith Moorthy

import tensorflow as tf

from get_data import import_data_movie
import matplotlib.pyplot as plt
import math
import numpy as np
import envoy

import scipy.sparse
import progressbar
import h5py
nusers = 458293
nitems = 17770
old_model = False
batch_size = 100 # divisors of nitems: {1, 2, 5, 10, 1777, 3554, 8885, 17770}
num_epochs = 30

# hyperparameters
latent_vars = 100
learning_rate = 0.001 #learned 0.1 and 0.5 are too high; 0.01 may be right
beta_reg = 0.001

datafile = h5py.File('../mu/all_baselined_bellkor96.h5')
resid_data = h5py.File('../mu/svd_reduced_ratings')
users = datafile['train_user_list']
items = datafile['train_item_list']

ratings = resid_data['train_rating_list']

probe_users = datafile['probe_user_list']
probe_items = datafile['probe_item_list']

probe_ratings = resid_data['probe_rating_list']
print len(ratings), len(probe_ratings)


print "loaded all data"

datatr = scipy.sparse.csr_matrix((ratings, (items, users)), shape=(nitems, nusers))

datapr = scipy.sparse.csr_matrix((probe_ratings, (probe_items, probe_users)), shape=(nitems, nusers))



print 'put into scipy sparse mat'
# Set seeds for random numbers to ensure reproducibility of results
np.random.seed(11261996)
tf.set_random_seed(11261996)


if old_model:
    autoRecSession = tf.Session()
    saver = tf.train.import_meta_graph('../AutoRec.mdl.meta')
    saver.restore(autoRecSession, tf.train.latest_checkpoint('../'))
    trainStep = tf.get_collection('trainStep')[0]
    squared_error_tf = tf.get_collection('squared_error_tf')[0]
else:
    # We create placeholders to correctly implementing the 'masking' feature
    # When training, only want to update the weights corresponding to observed param
    # This is the hardest part
    train_data = tf.placeholder(tf.float32, [None, nusers], name="train_data")
    train_observations = tf.placeholder(tf.float32, [None, nusers], name="train_observations") # the 'mask'

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

    test_data = tf.placeholder(tf.float32, [None, nusers], name="test_data")
    test_observations = tf.placeholder(tf.float32, [None, nusers], name="test_observations")
    squared_error_tf = tf.reduce_sum(tf.square((f - test_data)*test_observations))

    tf.add_to_collection('predictor', f)
    tf.add_to_collection('squared_error_tf', squared_error_tf)
    # loss is masked to only include observed
    masked_loss = tf.reduce_mean(tf.reduce_sum(tf.square((f - train_data)*train_observations), 1, keep_dims=True))
    # add regularization
    regularizers = tf.nn.l2_loss(W) + tf.nn.l2_loss(V)
    masked_loss = tf.reduce_mean(masked_loss + beta_reg *regularizers)
    trainStep = tf.train.RMSPropOptimizer(learning_rate).minimize(masked_loss)
    tf.add_to_collection('trainStep', trainStep)
    # start tf session and training
    saver = tf.train.Saver()
    autoRecSession = tf.InteractiveSession()
    autoRecSession.run(tf.global_variables_initializer())

# go through epochs
# randomly iterating through items
# it is found in the paper that items are better to put as data points instead of users
items_list = np.array(range(nitems))

print "Starting training epochs"
probe_rmse_lst = []
train_rmse_lst = []
datapr_num = len(probe_ratings)
datatr_num = len(ratings)
try:
    for epoch in range(num_epochs):
        print "Epoch %d of %d" % (epoch +1, num_epochs)
        np.random.shuffle(items_list)
        batch_num = int(math.ceil(nitems / float(batch_size)))

        bar = progressbar.ProgressBar(maxval=batch_num, widgets=["Training: ",
                                                                 progressbar.Bar(
                                                                     '=', '[', ']'),
                                                                 ' ', progressbar.Percentage(),

                                                                 ' ', progressbar.ETA()]).start()
        for batch in range(batch_num):

            if (batch % (math.ceil(0.01*batch_num))) == 0:
                bar.update(batch % bar.maxval)

            start_idx = batch * batch_size
            end_idx = min(start_idx + batch_size, nitems)
            #print batch, start_idx, end_idx

            batch_items = items_list[start_idx:end_idx]
            batch_data = datatr[batch_items, :]


            batch_observations = (batch_data != 0).toarray()

            batch_data = batch_data.toarray()



            trainStep.run(session = autoRecSession, feed_dict={"train_data:0":batch_data, "train_observations:0":batch_observations})
        bar.finish()
        if epoch % 5 == 0:
            #saver.save(autoRecSession, 'AutoRec'+(epoch+1)+'.mdl')
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

                if (batch % (math.ceil(0.01*batch_num))) == 0:
                    bar.update(batch % bar.maxval)

                start_idx = batch * batch_size
                end_idx = min(start_idx + batch_size, nitems)
                #print batch, start_idx, end_idx
                datapr_all = datapr[start_idx:end_idx, :]
                datatr_all = datatr[start_idx:end_idx, :]
                datatr_observation = (datatr_all != 0).toarray()
                datapr_observation = (datapr_all != 0).toarray()


                datapr_all = datapr_all.toarray()
                datatr_all = datatr_all.toarray()

                probe_squared_error += squared_error_tf.eval(session = autoRecSession, feed_dict={"train_data:0":datatr_all, "test_data:0":datapr_all, "test_observations:0":datapr_observation})
                train_squared_error += squared_error_tf.eval(session = autoRecSession,feed_dict={"train_data:0":datatr_all, "test_data:0":datatr_all, "test_observations:0":datatr_observation})

            bar.finish()
            probe_rmse_lst.append(math.sqrt(probe_squared_error/datapr_num))
            train_rmse_lst.append(math.sqrt(train_squared_error/datatr_num))
            print "Finished epoch %d of %d with train rmse of %.4f and probe rmse of %.4f" % (epoch+1, num_epochs, train_rmse_lst[-1],probe_rmse_lst[-1])

except KeyboardInterrupt:
    print 'saving progress and closing'

saver.save(autoRecSession, '../AutoRec.mdl')

f = plt.figure(1)
# take negative to get lower bound
plt.plot(range(1,len(probe_rmse_lst)+1), probe_rmse_lst, label='Probe_rmse')

plt.xlabel('Epoch')
plt.legend()
plt.ylim((.9,1))
plt.ylabel('I-AutoRec RMSE')

f.savefig('i_autorec_rmse.pdf')

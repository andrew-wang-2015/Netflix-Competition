import tensorflow as tf
import math
import numpy as np
import scipy.sparse
import progressbar
import h5py
nusers = 458293
nitems = 17770
batch_size = 100

datafile = h5py.File('../mu/all.h5')

users = datafile['train_user_list']
items = datafile['train_item_list']

ratings = datafile['train_rating_list']

probe_users = datafile['probe_user_list']
probe_items = datafile['probe_item_list']

probe_ratings = datafile['probe_rating_list']

batch_num = int(math.ceil(nitems / float(batch_size)))

print "loaded all data"

datatr = scipy.sparse.csr_matrix((ratings, (items, users)), shape=(nitems, nusers))

datapr = scipy.sparse.csr_matrix((probe_ratings, (probe_items, probe_users)), shape=(nitems, nusers))


sess = tf.Session()
new_saver = tf.train.import_meta_graph('../AutoRec.mdl.meta')
new_saver.restore(sess, tf.train.latest_checkpoint('../'))
squared_error_tf = tf.get_collection('squared_error_calc')[0]

probe_squared_error = 0.0
train_squared_error = 0.0
datapr_num = len(probe_ratings)
datatr_num = len(ratings)
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

    probe_squared_error += sess.run(squared_error_tf, feed_dict={"train_data:0":datatr_all, "test_data:0":datapr_all, "test_observations:0":datapr_observation})
    train_squared_error += sess.run(squared_error_tf, feed_dict={"train_data:0":datatr_all, "test_data:0":datatr_all, "test_observations:0":datatr_observation})

bar.finish()
print math.sqrt(probe_squared_error/datapr_num)
print math.sqrt(train_squared_error/datatr_num)

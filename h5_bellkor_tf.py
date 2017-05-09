import tensorflow as tf

scale = math.sqrt(6.0 / (nusers+ndays))
bu = tf.Variable(tf.random_uniform([nusers], -1, 1))
au = tf.Variable(tf.random_uniform([nusers], -1, 1))
but = tf.Variable(tf.random_uniform([nusers, ndays], -1, 1))
bi = tf.Variable(tf.random_uniform([nitems], -1, 1))
bibin = lil_matrix((nitems, nbins), dtype ='f')
cu = tf.Variable(tf.random_uniform([nusers], 0, 2))
cut = lil_matrix((nusers, ndays), dtype ='f')
mean_times = (R_u_t.sum(axis = 1).flatten())/R_u_t.getnnz(axis = 1)

diffut = all_d - mean_times[:,all_I]
devut = (np.multiply(np.sign(diffut), np.power(abs(diffut), beta)))


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

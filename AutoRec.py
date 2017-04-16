# Implementation of AutoRec in tf. See http://users.cecs.anu.edu.au/~u5098633/papers/www15.pdf
# Aadith Moorthy
import easygui
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2, l1
from get_data import import_data_to_mat_user_opt
from keras.optimizers import Adagrad
import math
import numpy as np
import envoy
import random


nusers = 7#458293
nitems = 17770

batch_size = 1 # divisors of nusers: {1, 11, 61, 671, 683, 7513, 41663, 458293}
num_epochs = 1

# hyperparameters
latent_vars = 50
learning_rate = 0.1

trainpath = '../um/small_train.dta'
probepath = '../um/small_train.dta'
datatr, datapr = import_data_movie(trainpath, probepath)

# Set seeds for random numbers to ensure reproducibility of results
np.random_seed(11261996)
tf.set_random_seed(11261996)
random.seed(11261996)

# get proper testing data ready
datats_in = []
datats_out = []
datats_obs = []
for i in datapr

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
μ = tf.Variable(tf.random_uniform([latent_vars], -scale, scale))
g = tf.nn.softmax(tf.matmul(train_data, V) + μ)
#activation here can be changed as needed

# layer 2
W = tf.Variable(tf.random_uniform([latent_vars, nusers], -scale, scale))
b = tf.Variable(tf.random_uniform([nusers], -scale, scale))
f = tf.matmul(train_data, W) + b #no activation because ratings are -2,-1,0,1,2

test_data = tf.placeholder(tf.float32, [None, nusers])
test_observations = tf.placeholder(tf.float32, [None, nusers])
root_mean_squared_error = tf.sqrt(tf.reduce_sum(tf.square((y - test_data)*test_observations)) / tf.reduce_sum(test_observations))

# start tf session and training
autoRecSession = tf.InteractiveSession()
autoRecSession.run(tf.initialize_all_variables())

# loss is masked to only include observed
masked_loss = tf.reduce_mean(tf.reduce_sum(tf.square((y - train_data)*train_observations), 1, keep_dims=True))
trainStep = tf.train.GradientDescentOptimizer(learning_rate).minimize(masked_loss)

# go through epochs
# randomly iterating through items
# it is found in the paper that items are better to put as data points instead of users
items_list = range(nitems)

print "Starting training epochs"
for epoch in range(num_epochs):
    print "Epoch %d of %d" % (epoch, num_epochs)
    random.shuffle(items_list)

    for batch in range(math.ceil(nitems / batch_size)):
        start_idx = batch * batchSize
        end_idx = min(start_idx + batchSize, nitems)

        batch_data = []
        batch_observations = []
        for item_idx in range(start_idx, end_idx):
            item = items_list[item_idx]
            item_features = datatr.getrow(item)

            batchData.append(item_features.toarray())
            item_observation = np.zeros((1, nusers))
            item_observation[item_features.indices] = 1
            batch_observations.append(item_observation)

        batch_data = np.array(batch_data)
        batch_observations = np.array(batch_observations)

        trainStep.run(feed_dict={train_data:batch_data, train_observations:batch_observations})

    result = rmse.eval(feed_dict={train_data:allData, preData:allTestData, preMask:allTestMask})
    print("epoch %d/%d\trmse: %.4f"%(epoch+1, epochCount, result))



'''
class Adagrad_select(Adagrad):


    def get_updates(self, params, constraints, loss):
        res = super(Adagrad_select, self).get_updates(params, constraints, loss)

        #print tf.Session().run(res), 'sf'
        res[0] = tf.matmul(tf.convert_to_tensor(np.zeros((50, 17770)), dtype='float32'), res[0])
        res[1] = tf.matmul(tf.convert_to_tensor(np.zeros((50, 17770)), dtype='float32'), res[1])

        return res

def init_model():
    model = Sequential()

    # latent variables
    model.add(Dense(50, activation='sigmoid', input_shape=(nitems, )))

    #out
    model.add(Dense(nitems, activation='linear'))

    model.compile(optimizer=Adagrad_select(),
              loss='mse')
    return model

def data_generator():
    print range(0, nusers, batch_size)
    while True:
        for i in range(0, nusers, batch_size):
            data_for_batch = datatr[i:i+batch_size, :].toarray()
            print data_for_batch, 'd'
            yield data_for_batch, data_for_batch

def probe_mse_calc(model, probepath):
    f = open(probepath, 'r')
    mean_squared_error = 0
    r = envoy.run('wc -l {}'.format(probepath))
    num_lines = int(r.std_out.strip().partition(' ')[0])
    for line in f:
        user, item, date, rating = line.split()

        mean_squared_error += ((model.predict(datatr[int(user)-1].toarray())[:,int(item)-1] - (float(rating)-3))**2)/float(num_lines)
    f.close()
    return mean_squared_error

if __name__=="__main__":

    model = init_model()

    model.fit_generator(data_generator(), samples_per_epoch=nusers, nb_epoch=nepochs)
    print math.sqrt(probe_mse_calc(model, probepath))
'''

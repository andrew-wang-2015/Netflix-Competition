from keras.layers import Embedding, Reshape, Merge
from keras.models import Sequential
from keras.optimizers import Adamax
import envoy
import progressbar
import numpy as np
import h5py
import math
import random
h5filename = "../um/all.h5"

h5file = h5py.File(h5filename)

I = h5file['train_user_list']
J = h5file['train_item_list']
V = h5file['train_rating_list']

probe_I = h5file['probe_user_list']
probe_J = h5file['probe_item_list']
probe_V = h5file['probe_rating_list']

numUsers = 458293
numMovies = 17770
dataPoints = len(I)
batch_size= 10000 # {1, 7, 31, 217, 452957, 3170699, 14041667, 98291669}
num_epochs = 10

factors = 20
left = Sequential()
left.add(Embedding(numUsers, factors,input_length=1))
left.add(Reshape((factors,)))
right = Sequential()
right.add(Embedding(numMovies, factors, input_length=1))
right.add(Reshape((factors,)))
model = Sequential()
model.add(Merge([left, right], mode='dot'))
model.compile(loss='mse', optimizer='rmsprop')
print 'declared model'

print 'fitting model'
for epoch in range(num_epochs):
    print "Epoch: ", epoch
    bar = progressbar.ProgressBar(maxval=int(math.ceil(dataPoints/float(batch_size))), widgets=["Training: ",
                                                             progressbar.Bar(
                                                                 '=', '[', ']'),
                                                             ' ', progressbar.Percentage(),

                                                             ' ', progressbar.ETA()]).start()
    for batch_num in range(int(math.ceil(dataPoints/float(batch_size)))):
        if (random.random() < .1):
            if (batch_num % 1) == 0:
                bar.update(batch_num % bar.maxval)
            start = batch_num * batch_size
            end = min((batch_num+1) * batch_size, dataPoints)

            model.train_on_batch([I[start:end], J[start:end]], V[start:end])
    bar.finish()
    print model.evaluate([probe_I[:], probe_J[:]], probe_V[:]), 'loss on test'

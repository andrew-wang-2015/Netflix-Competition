from keras.layers import Embedding, Reshape, Merge, Dropout, Dense
from keras.models import Sequential, load_model
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
import progressbar
import numpy as np
import h5py
import math
import random
import time

h5filename = "../mu/all_baselined_bellkor96.h5"

h5file = h5py.File(h5filename)

I = h5file['train_user_list']
J = h5file['train_item_list']
V = h5file['train_rating_list']
d = h5file['train_time_list']


probe_I = h5file['probe_user_list']
probe_J = h5file['probe_item_list']
probe_V = h5file['probe_rating_list']
probe_d = h5file['probe_time_list']

qual_I = h5file['qual_user_list']
qual_J = h5file['qual_item_list']
qual_d = h5file['qual_time_list']

numUsers = 458293
numMovies = 17770
numDays = 2243
dataPoints = len(I)

batch_size= 100000 # {1, 7, 31, 217, 452957, 3170699, 14041667, 98291669}
num_epochs = 30

factors = 100
dropouts = [0.2]
best_loss = 1
best_dropout = 0
for drop in dropouts:
    try:
        model = load_model('svd_keras0.2wewe.mdl')
    except:
        left = Sequential()
        left.add(Embedding(numUsers, factors,input_length=1))
        left.add(Reshape((factors,)))
        right = Sequential()
        right.add(Embedding(numMovies, factors, input_length=1))
        right.add(Reshape((factors,)))
        time = Sequential()
        time.add(Embedding(numDays, factors, input_length=1))
        time.add(Reshape((factors,)))
        model = Sequential()
        model.add(Merge([left, right, time], mode='concat'))
        model.add(Dropout(drop*1.5))
        model.add(Dense(factors, activation='relu'))
        model.add(Dropout(drop))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse', optimizer='rmsprop')
    print 'declared model'

    print 'fitting model'

    '''for epoch in range(num_epochs):
        print "Epoch: ", epoch
        bar = progressbar.ProgressBar(maxval=int(math.ceil(dataPoints/float(batch_size))), widgets=["Training: ",
                                                                 progressbar.Bar(
                                                                     '=', '[', ']'),
                                                                 ' ', progressbar.Percentage(),

                                                                 ' ', progressbar.ETA()]).start()
        for batch_num in range(int(math.ceil(dataPoints/float(batch_size)))):

            if (batch_num % 1) == 0:
                bar.update(batch_num % bar.maxval)
            start = batch_num * batch_size
            end = min((batch_num+1) * batch_size, dataPoints)

            model.train_on_batch([I[start:end], J[start:end]], V[start:end])
        bar.finish()'''



    callbacks = [EarlyStopping('val_loss', patience=2),
             ModelCheckpoint('svd_keras'+str(drop)+'.mdl', save_best_only=True)]
    model.fit([I,J, d], V, batch_size=batch_size, nb_epoch = num_epochs, validation_data=([probe_I, probe_J, probe_d], probe_V), callbacks=callbacks)

    del model
    model = load_model('svd_keras'+str(drop)+'.mdl')
    loss =  model.evaluate([probe_I,probe_J, probe_d], probe_V, batch_size=batch_size)

    if loss < best_loss:
        best_loss = loss
        best_dropout = drop


print 'the best loss and droupout were', best_loss, best_dropout
model = load_model('svd_keras'+str(best_dropout)+'.mdl')
print 'loss on probe', model.evaluate([probe_I,probe_J, probe_d], probe_V, batch_size=batch_size)
pred = model.predict([qual_I,qual_J, qual_d], batch_size=batch_size)
print len(pred), len(qual_I)
pred_file = open('predictions_untransformed.txt', 'w')
np.savetxt(pred_file, pred)

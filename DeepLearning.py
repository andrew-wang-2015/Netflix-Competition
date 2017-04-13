movie_count = 17771
user_count = 2649430
model_left = Sequential()
model_left.add(Embedding(movie_count, 60, input_length=1))
model_right = Sequential()
model_right.add(Embedding(user_count, 20, input_length=1))
model = Sequential()
model.add(Merge([model_left, model_right], mode='concat'))
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('sigmoid'))
model.add(Dense(64))
model.add(Activation('sigmoid'))
model.add(Dense(64))
model.add(Activation('sigmoid'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adadelta')
model.fit([tr[:,0].reshape((L,1)), tr[:,1].reshape((L,1))], 
          tr[:,2].reshape((L,1)), batch_size=24000, nb_epoch=42, 
          validation_data=([ ts[:,0].reshape((M,1)), 
                             ts[:,0].reshape((M,1))], ts[:,2].reshape((M,1))))

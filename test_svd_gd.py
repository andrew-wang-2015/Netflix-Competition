
from keras.models import load_model
import math
import numpy as np
import h5py, progressbar
h5filename = "../um/all_baselined.h5"

h5file = h5py.File(h5filename)
model = load_model('../svd_keras.mdl')
probe_I = h5file['probe_user_list']
probe_J = h5file['probe_item_list']
probe_V = h5file['probe_rating_list']
qual_I = h5file['qual_user_list']
qual_J = h5file['qual_item_list']

dataPoints = len(probe_I)
batch_size = 1
probe_se = 0.0
bar = progressbar.ProgressBar(maxval=int(math.ceil(dataPoints/float(batch_size))), widgets=["Testing: ",
                                                         progressbar.Bar(
                                                             '=', '[', ']'),
                                                         ' ', progressbar.Percentage(),

                                                         ' ', progressbar.ETA()]).start()

model.evaluate([probe_I, probe_J], probe_V)

'''for batch_num in range(int(math.ceil(dataPoints/float(batch_size)))):

    if (batch_num % 1000) == 0:
        bar.update(batch_num % bar.maxval)
    start = batch_num * batch_size
    end = min((batch_num+1) * batch_size, dataPoints)

    preding = np.array(model.predict_on_batch([probe_I[start:end], probe_J[start:end]]))
    given = probe_V[start:end]
    #print preding[0:4,:], 'sdfs'
    #print model.predict_on_batch([probe_I[start:start+4], probe_J[start:start+4]]), 'sdf'
    probe_se += np.sum(np.power((given-preding),2))
    #print probe_se/((batch_num+1) * batch_size)'''

bar.finish()
print math.sqrt(probe_se/dataPoints)

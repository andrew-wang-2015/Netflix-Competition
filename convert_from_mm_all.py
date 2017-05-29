import scipy.io as sio
from scipy.sparse import coo_matrix, csr_matrix
import numpy as np
import h5py
import envoy
import progressbar

source_filename = '../virtualbox_share/all_data.mm.predictontrain'

r = envoy.run('wc -l {}'.format(source_filename))
#num_lines = int(r.std_out.strip().partition(' ')[0])
bar = progressbar.ProgressBar(maxval=100000000, widgets=["Loading train ratings: ",
                                                         progressbar.Bar(
                                                             '=', '[', ']'),
                                                         ' ', progressbar.Percentage(),

                                                         ' ', progressbar.ETA()]).start()

all_I = np.empty(99666408)
all_J = np.empty(99666408)
all_np = np.empty(99666408)
with open(source_filename) as f:
	for i, line in enumerate(f):
		if i % 100000 == 0:
			bar.update(i % bar.max_value)
		if i > 2:
			line = line.split()
			all_I[i - 3] = int(line[0])
			all_J[i - 3] = int(line[1])
			all_np[i - 3] = float(line[2])
			#all_rat.append(line[2])
bar.finish()

print('Read in All Predictions')


#all_np = np.array(all_rat)
print(all_np.shape)
print (all_np[0])
print (all_np[1])
print (all_np[2])
ind = np.lexsort((all_I, all_J))
print (ind)
print (all_I[ind[0]], all_J[ind[0]], all_np[ind[0]])
print (all_I[ind[1]], all_J[ind[1]], all_np[ind[1]])
print (all_I[ind[2]], all_J[ind[2]], all_np[ind[2]])
all_np = all_np[ind]
print (all_np[0])
print (all_np[1])
print (all_np[2])

#print('User: ', all_I[0], 'Movie:', all_J[0],'Rating:', all_V[0])

dest_filename = '../mu/rbm_all_on_train.h5'

f = h5py.File(dest_filename, 'w')
#f.create_dataset('train_user_list', data = all_np[:,0])
#f.create_dataset('train_item_list', data = all_np[:,1])
f.create_dataset('train_predictions', data = all_np)
#f.create_dataset('qual_user_list', data = qual_np[:,0])
#f.create_dataset('qual_item_list', data = qual_np[:,1])
#f.create_dataset('qual_predictions', data = qual_np[:,2])
f.close()
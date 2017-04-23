import envoy
import progressbar
import numpy as np
import h5py
import scipy.sparse

nusers = 458293
nitems = 17770
source_filename = "../mu/all.dta"
idx_filename = "../mu/all.idx"
dest_filename = '../mu/all_RBM.h5'

r = envoy.run('wc -l {}'.format(source_filename))
num_lines = int(r.std_out.strip().partition(' ')[0])
bar = progressbar.ProgressBar(maxval=num_lines, widgets=["Loading train ratings: ",
                                                         progressbar.Bar(
                                                             '=', '[', ']'),
                                                         ' ', progressbar.Percentage(),

                                                         ' ', progressbar.ETA()]).start()
I = []
J = []
V = []
probe_I = []
probe_J = []
probe_V = []
qual_I = []
qual_J = []
RBM_rows = []

with open(source_filename) as f:
      idx_file = open(idx_filename)
      for i, line in enumerate(f):

          if (i % 1000) == 0:
              bar.update(i % bar.max_value)

          userid, itemid, date, rating = line.split()


          idx = int(idx_file.readline().replace("\n",""))
          if idx <= 3:

              I.append(int(userid)-1)
              #print I
              J.append(int(itemid)-1)
              V.append(int(rating)) # to get mean
              RBM_rows.append((int(userid)-1) * 5 + int(rating))
          elif idx == 4:
              probe_I.append(int(userid)-1)
              probe_J.append(int(itemid)-1)
              probe_V.append(int(rating))
          elif idx == 5:
              qual_I.append(int(userid)-1)
              qual_J.append(int(itemid)-1)

bar.finish()
print ("Finished Bar: ")
I = np.array(I)
print ("Finished 1: ")
J = np.array(J)
print ("Finished 2: ")
V = np.array(V)
print ("Finished 3: ")
probe_I = np.array(probe_I)
print ("Finished 4: ")
probe_J = np.array(probe_J)
print ("Finished 5: ")
probe_V = np.array(probe_V)
print ("Finished 6: ")
qual_I = np.array(qual_I)
print ("Finished 7: ")
qual_J = np.array(qual_J)
print ("Finished 8: ")
RBM_rows = np.array(RBM_rows)


print ("starting writes")
f = h5py.File(dest_filename, 'w')
f.create_dataset('train_user_list', data = I)
f.create_dataset('train_item_list', data = J)
f.create_dataset('train_rating_list', data = V)
f.create_dataset('probe_user_list', data = probe_I)
f.create_dataset('probe_item_list', data = probe_J)
f.create_dataset('probe_rating_list', data = probe_V)
f.create_dataset('qual_user_list', data = qual_I)
f.create_dataset('qual_item_list', data = qual_J)
f.create_dataset('train_RBM_rows_list', data = RBM_rows)


print ('finished single arrays')

f.close()

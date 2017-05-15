import envoy
import progressbar
import numpy as np
import h5py
import scipy.sparse

nusers = 458293
nitems = 17770
source_filename = "../mu/all.dta"
idx_filename = "../mu/all.idx"
dest_filename = '../mu/all_postprocess.h5'

#r = envoy.run('wc -l {}'.format(source_filename))
#num_lines = int(r.std_out.strip().partition(' ')[0])
bar = progressbar.ProgressBar(maxval=102416306, widgets=["Loading all ratings: ",
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

with open(source_filename) as f:
      idx_file = open(idx_filename)
      for i, line in enumerate(f):

          if (i % 1000) == 0:
              bar.update(i % bar.max_value)

          userid, itemid, date, rating = line.split()


          idx = int(idx_file.readline().replace("\n",""))
          if idx <= 4:

              I.append(int(userid)-1)
              #print I
              J.append(int(itemid)-1)
              V.append(int(rating)) # to get mean

bar.finish()
I = np.array(I)
J = np.array(J)
V = np.array(V)

print ("starting writes")
f = h5py.File(dest_filename, 'w')
f.create_dataset('all_user_list', data = I)
f.create_dataset('all_item_list', data = J)
f.create_dataset('all_rating_list', data = V)


print ('finished single arrays')

f.close()

import envoy
import progressbar
from scipy.sparse import csr_matrix
import numpy as np
import h5py
from sklearn.externals import joblib
from sklearn.linear_model import SGDRegressor

# baselining as per http://blog.echen.me/2011/10/24/winning-the-netflix-prize-a-summary/
nusers = 458293
nitems = 17770
source_filename = "../mu/all.dta"
idx_filename = "../mu/all.idx"
dest_filename = '../mu/all_baselined_time.h5'
baseline_fitter_filename = '../mu/sgd_fitter.pkl'
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
d = []
probe_I = []
probe_J = []
probe_V = []
probe_d = []
qual_I = []
qual_J = []
qual_d = []

with open(source_filename) as f:
      idx_file = open(idx_filename)
      for i, line in enumerate(f):
          if (i % 1000) == 0:
              bar.update(i % bar.maxval)

          userid, itemid, date, rating = line.split()


          idx = int(idx_file.readline().replace("\n",""))
          if idx <= 3:

              I.append(int(userid)-1)
              #print I
              J.append(int(itemid)-1)
              V.append(int(rating)-3-0.60951619727280626) # to get mean
              d.append(int(date))
          elif idx == 4:
              probe_I.append(int(userid)-1)
              probe_J.append(int(itemid)-1)
              probe_V.append(int(rating)-3-0.60951619727280626)
              probe_d.append(int(date))
          elif idx == 5:
              qual_I.append(int(userid)-1)
              qual_J.append(int(itemid)-1)
              qual_d.append(int(date))

bar.finish()
I = np.array(I)
J = np.array(J)
V = np.array(V)
d = np.array(d)
probe_I = np.array(probe_I)
probe_J = np.array(probe_J)
probe_V = np.array(probe_V)
probe_d = np.array(probe_d)
qual_I = np.array(qual_I)
qual_J = np.array(qual_J)
qual_d = np.array(qual_d)

# note: later we must add the probe to this and train
all_V = V#np.concatenate((V, probe_V))
all_I = I#np.concatenate((I, probe_I))
all_J = J#np.concatenate((J, probe_J))
all_d = np.concatenate((d, probe_d))
print 'done loading single arrays'
R_m = csr_matrix((all_V,(all_J,all_I)), shape=(nitems, nusers))

R_u = csr_matrix((all_V,(all_I,all_J)), shape=(nusers, nitems))
R_u_t = csr_matrix((all_d,(np.concatenate((I, probe_I)),np.concatenate((J, probe_J)))), shape=(nusers, nitems))

print 'done building scipy sparse matrices'
movie_avg = np.zeros(nitems)
batch_size = 10000
for i in range(0, nitems, batch_size):
    movie_current = R_m[i:min(i+batch_size, nitems), :]
    batch_avg = ((movie_current.sum(axis =1).flatten())/movie_current.getnnz(axis =1))
    movie_avg[i:min(i+batch_size, nitems)] = batch_avg

user_avg = np.zeros(nusers)

for i in range(0, nusers, batch_size):
    users_current = R_u[i:min(i+batch_size, nusers), :]
    batch_avg = ((users_current.sum(axis =1).flatten())/users_current.getnnz(axis =1))
    user_avg[i:min(i+batch_size, nusers)] = batch_avg

print 'done avging', movie_avg, user_avg

n_epochs = 10
print 'building baseline SGD predictor, training for', n_epochs, 'epochs'
lin_model = SGDRegressor()

rat_num = len(all_I)
for e in range(n_epochs):
    for i in range(0, rat_num, batch_size):

        given = all_V[i:min(i+batch_size, rat_num)]
        u_mean = user_avg[all_I[i:min(i+batch_size, rat_num)]]
        m_mean = movie_avg[all_J[i:min(i+batch_size, rat_num)]]
        times = np.log(R_u_t[all_I[i:min(i+batch_size, rat_num)],all_J[i:min(i+batch_size, rat_num)]])

        u_mean = np.array([u_mean]).T
        m_mean = np.array([m_mean]).T
        times = times.T

        preding = np.concatenate((u_mean, m_mean, times), axis = 1)

        lin_model.partial_fit(preding, given)


joblib.dump(lin_model, baseline_fitter_filename)

print 'incorporating learned deviations'
num_rats = len(I)
for i in range(0, num_rats, batch_size):
    u_mean = user_avg[I[i:min(i+batch_size, num_rats)]]
    m_mean = movie_avg[J[i:min(i+batch_size, num_rats)]]
    times = np.log(R_u_t[I[i:min(i+batch_size, num_rats)],J[i:min(i+batch_size, num_rats)]])
    u_mean = np.array([u_mean]).T
    m_mean = np.array([m_mean]).T
    times = times.T
    V[i:min(i+batch_size, num_rats)] = V[i:min(i+batch_size, num_rats)] - lin_model.predict(np.concatenate((u_mean, m_mean, times), axis = 1))

num_rats = len(probe_I)
for i in range(0, num_rats, batch_size):
    u_mean = user_avg[probe_I[i:min(i+batch_size, num_rats)]]
    m_mean = movie_avg[probe_J[i:min(i+batch_size, num_rats)]]
    times = np.log(R_u_t[probe_I[i:min(i+batch_size, num_rats)],probe_J[i:min(i+batch_size, num_rats)]])
    u_mean = np.array([u_mean]).T
    m_mean = np.array([m_mean]).T
    times = times.T
    probe_V[i:min(i+batch_size, num_rats)] = probe_V[i:min(i+batch_size, num_rats)]  - lin_model.predict(np.concatenate((u_mean, m_mean, times), axis = 1))



print "starting writes"
f = h5py.File(dest_filename, 'w')
f.create_dataset('train_user_list', data = I)
f.create_dataset('train_item_list', data = J)
f.create_dataset('train_rating_list', data = V)
f.create_dataset('train_time_list', data = d)
f.create_dataset('train_user_avg', data = user_avg)
f.create_dataset('train_item_avg', data = movie_avg)
f.create_dataset('probe_user_list', data = probe_I)
f.create_dataset('probe_item_list', data = probe_J)
f.create_dataset('probe_rating_list', data = probe_V)
f.create_dataset('probe_time_list', data = probe_d)
f.create_dataset('qual_user_list', data = qual_I)
f.create_dataset('qual_item_list', data = qual_J)
f.create_dataset('qual_time_list', data = qual_d)


print 'finished single arrays'

f.close()

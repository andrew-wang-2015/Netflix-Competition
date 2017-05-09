import numpy as np
import h5py
import gnumpy as gpu
from scipy.sparse import coo_matrix, csr_matrix
import math
datafile = h5py.File('../mu/all_baselined_bellkor96.h5')


qual_I = datafile['qual_user_list'][:]#np.array(qual_I)
qual_J = datafile['qual_item_list'][:]#np.array(qual_J)
qual_d = datafile['qual_time_list'][:]#np.array(qual_d)'''
#probe_V = datafile['probe_rating_list'][:]#
probe_test = h5py.File('../mu/all.h5')['probe_rating_list']
all_d = datafile['train_time_list'][:]
all_I = datafile['train_user_list'][:]
all_J = datafile['train_item_list'][:]

bu = datafile['bu'][:]
au = datafile['au'][:]
cu = datafile['cu'][:]
bi = datafile['bi'][:]
beta = 0.4
nusers = 458293
nitems = 17770
ndays = 2243
nbins = 30.0
R_u_t = csr_matrix((all_d,(all_I,all_J)), shape=(nusers, nitems))
but = coo_matrix((datafile['butd'][:], (datafile['butr'][:], datafile['butc'][:])), shape=(nusers, ndays)).tolil()
cut = coo_matrix((datafile['cutd'][:], (datafile['cutr'][:], datafile['cutc'][:])), shape=(nusers, ndays)).tolil()
bibin = coo_matrix((datafile['bibind'][:], (datafile['bibinr'][:], datafile['bibinc'][:])), shape=(nitems, nbins)).tolil()
# TODO fix proper time getting
mean_times = (R_u_t.sum(axis = 1).flatten())/R_u_t.getnnz(axis = 1)

diffut = qual_d - mean_times[:,qual_I]
devut = (np.multiply(np.sign(diffut), np.power(abs(diffut), beta)))

batch_size = 100000

#print 'oll', math.sqrt(np.sum(np.power(probe_V, 2))/len(probe_V))




u_pred = np.loadtxt('predictions_untransformed.txt')
print len(qual_J), len(u_pred)
pred = open('predictions.txt', 'w')
'''probe_I = qual_I
num_rats = len(probe_I)
probe_V_predtest = np.zeros(num_rats)


probe_J = qual_J
probe_d = qual_d
probe_devut = devut
preding = np.zeros(num_rats)
for i in range(0, num_rats, batch_size):
    current_users = probe_I[i:min(i+batch_size, num_rats)]
    list_lens = len(current_users)
    current_movies = probe_J[i:min(i+batch_size, num_rats)]
    current_days = probe_d[i:min(i+batch_size, num_rats)]
    current_bins = np.floor(current_days/(ndays/nbins))
    current_bu = gpu.garray(bu[current_users])
    current_au = gpu.garray(au[current_users])
    current_devut = gpu.garray(probe_devut[:, i:min(i+batch_size, num_rats)])
    current_but = gpu.garray(but[current_users, current_days].toarray())
    current_bi = gpu.garray(bi[current_movies])
    current_bibin = gpu.garray(bibin[current_movies, current_bins].toarray())
    current_cu = gpu.garray(cu[current_users])
    current_cut = gpu.garray(cut[current_users, current_days].toarray())

    pred = -(- current_bu- current_but -current_au * current_devut -(current_bi+ current_bibin)*(current_cu + current_cut))#
    preding[i:min(i+batch_size, num_rats)] = pred
    probe_V_predtest[i:min(i+batch_size, num_rats)] = (probe_V[i:min(i+batch_size, num_rats)]+ pred).reshape(list_lens,)

prb = math.sqrt(np.sum(np.power(probe_test-probe_V_predtest, 2))/num_rats)
print 'should be 0 ', prb
print 'should be 0.959', math.sqrt(np.sum(np.power(probe_test-preding, 2))/num_rats)
print 'should be 0.959', math.sqrt(np.sum(np.power(probe_V, 2))/num_rats)
print np.sum(probe_test-preding == probe_V), probe_test-preding - probe_V
print 'as reported by prog', math.sqrt(np.sum(np.power(probe_V-u_pred, 2))/num_rats)'''
rat_num = len(qual_J)

#del datafile['probe_rating_list']
#datafile.create_dataset('probe_rating_list', data = probe_test-preding)
#datafile.close()



for i in range(0, rat_num, batch_size):
    current_users = qual_I[i:min(i+batch_size, rat_num)]
    list_lens = len(current_users)
    current_movies = qual_J[i:min(i+batch_size, rat_num)]
    current_days = qual_d[i:min(i+batch_size, rat_num)]
    current_bins = np.floor(current_days/(ndays/nbins))
    current_bu = gpu.garray(bu[current_users])
    current_au = gpu.garray(au[current_users])
    current_devut = gpu.garray(devut[:, i:min(i+batch_size, rat_num)])
    current_but = gpu.garray(but[current_users, current_days].toarray())
    current_bi = gpu.garray(bi[current_movies])
    current_bibin = gpu.garray(bibin[current_movies, current_bins].toarray())
    current_cu = gpu.garray(cu[current_users])
    current_cut = gpu.garray(cut[current_users, current_days].toarray())

    base_dev = -(- current_bu- current_but-current_au * current_devut- (current_bi+ current_bibin)*(current_cu + current_cut))

    preding = (u_pred[i:min(i+batch_size, rat_num)] + base_dev)+3.60951619727280626
    for pr in preding.T:
        pred.write('%.3f\n' % max(min(pr,5),1)) #convert back to 1-5 ratings systems
#print 'should be .91', math.sqrt(np.sum(np.power(probe_test-preding, 2))/len(probe_test))
#print np.sum(preding - base_devs)

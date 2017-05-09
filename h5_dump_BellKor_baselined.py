import envoy
import progressbar
from scipy.sparse import lil_matrix, csr_matrix, coo_matrix
import numpy as np
import h5py
from sklearn.externals import joblib
from sklearn.linear_model import SGDRegressor
import random
import gnumpy as gpu
import math
# baselining as per bellkor solution

# hyperparameters from paper
beta = 0.4
lr_bu = 3e-3
lr_but = 25e-4
lr_au = 1e-5
lr_bi = 2e-3
lr_bibin = 5e-5
lr_cu = 8e-3
lr_cut = 2e-3
reg_bu = 3e-2
reg_but = 5e-3
reg_au = 50
reg_bi = 3e-2
reg_bibin = 0.1
reg_cu = 0.01
reg_cut = 5e-3


nusers = 458293
nitems = 17770
ndays = 2243
f_inp = h5py.File('../mu/all.h5')
f_restart = h5py.File('../mu/all_baselined_bellkor962.h5')
dest_filename = '../mu/all_baselined_bellkor.h5'


num_lines = 102416306

#I = []
#J = []
#V = []
#d = []
#probe_I = []
#probe_J = []
#probe_V = []
#probe_d = []
#qual_I = []
#qual_J = []
#qual_d = []




I = f_inp['train_user_list'][:]#np.array(I)
J = f_inp['train_item_list'][:]#np.array(J)

V = f_inp['train_rating_list'][:]#np.array(V)
d = f_inp['train_time'][:]#np.array(d)
probe_I = f_inp['probe_user_list'][:]#np.array(probe_I)
probe_J = f_inp['probe_item_list'][:]#np.array(probe_J)
probe_V = f_inp['probe_rating_list'][:]#np.array(probe_V)
probe_d = f_inp['probe_time'][:]#np.array(probe_d)
qual_I = f_inp['qual_user_list'][:]#np.array(qual_I)
qual_J = f_inp['qual_item_list'][:]#np.array(qual_J)
qual_d = f_inp['qual_time'][:]#np.array(qual_d)

# note: later we must add the probe to this and train
all_V = V#np.concatenate((V, probe_V))
all_I = I#np.concatenate((I, probe_I))
all_J = J#np.concatenate((J, probe_J))
all_d = d#np.concatenate((d, probe_d))
print 'done loading single arrays'


R_u_t = csr_matrix((all_d,(all_I,all_J)), shape=(nusers, nitems))

print 'done building scipy sparse matrices'
print 'jj'

n_epochs = 5
nbins = 30.0 # from paper
'''bu = np.random.rand(nusers)*0.1-0.05
au = np.random.rand(nusers)*0.001
but = lil_matrix((nusers, ndays), dtype = 'f')
bi = np.random.rand(nusers)*0.1-0.05
bibin = lil_matrix((nitems, nbins), dtype ='f')
cu = np.ones(nusers)
cut = lil_matrix((nusers, ndays), dtype ='f')'''
bu = f_restart['bu'][:]
bi = f_restart['bi'][:]
au = f_restart['au'][:]
but = coo_matrix((f_restart['butd'][:], (f_restart['butr'][:], f_restart['butc'][:])), shape=(nusers, ndays)).tolil()
bibin = coo_matrix((f_restart['bibind'][:], (f_restart['bibinr'][:], f_restart['bibinc'][:])), shape=(nitems, nbins)).tolil()
cu = f_restart['cu'][:]
cut = coo_matrix((f_restart['cutd'][:], (f_restart['cutr'][:], f_restart['cutc'][:])), shape=(nusers, ndays)).tolil()
mean_times = (R_u_t.sum(axis = 1).flatten())/R_u_t.getnnz(axis = 1)

diffut = all_d - mean_times[:,all_I]
probe_diffut = probe_d - mean_times[:,probe_I]
devut = (np.multiply(np.sign(diffut), np.power(abs(diffut), beta)))
probe_devut = (np.multiply(np.sign(probe_diffut), np.power(abs(probe_diffut), beta)))

print 'building baseline SGD predictor, training for', n_epochs, 'epoch(s)'

rat_num = len(all_I)
min_prb = 1
dp_list = range(rat_num)
probe_V_test = np.zeros(len(probe_I))
for e in range(n_epochs):
    batch_size = 1000
    print "epoch: ", (e + 1)
    random.shuffle(dp_list)

    bar = progressbar.ProgressBar(maxval=rat_num, widgets=["SGD: ",
                                                             progressbar.Bar(
                                                                 '=', '[', ']'),
                                                             ' ', progressbar.Percentage(),

                                                             ' ', progressbar.ETA()]).start()
    for i in range(0, rat_num, batch_size):
        # useful variables
        bar.update(i % rat_num)
        current_idx = dp_list[i:min(i+batch_size, rat_num)]
        current_users = all_I[current_idx]
        list_lens = len(current_users)
        current_movies = all_J[current_idx]
        current_days = all_d[current_idx]
        current_bins = np.floor(current_days/(ndays/nbins))
        current_bu = gpu.garray(bu[current_users])
        current_au = gpu.garray(au[current_users])
        current_devut = gpu.garray(devut[:, current_idx])

        current_but = gpu.garray(but[current_users, current_days].toarray())


        current_bi = gpu.garray(bi[current_movies])
        current_bibin = gpu.garray(bibin[current_movies, current_bins].toarray())
        current_cu = gpu.garray(cu[current_users])
        current_cut = gpu.garray(cut[current_users, current_days].toarray())
        no_reg_inner = gpu.garray(all_V[current_idx]) - current_bu- current_but-(current_au*current_devut)- (current_bi+ current_bibin)*(current_cu + current_cut)

        #print no_reg_inner.shape, current_devut.shape
        # derivatives
        dldbu = -2*no_reg_inner+2*reg_bu*current_bu
        dldau = -2*no_reg_inner*current_devut+reg_au*2*current_au
        dldbut = -2*no_reg_inner+2*reg_but*current_but
        dldbi = -2*no_reg_inner*(current_cu + current_cut) +2 * reg_bi *current_bi
        dldbibin = -2*no_reg_inner* (current_cu + current_cut) +2 * reg_bibin * current_bibin
        dldcu = -2*no_reg_inner* (current_bi+ current_bibin) + 2* reg_cu * (current_cu-1)
        dldcut = -2*no_reg_inner*(current_bi+ current_bibin) + 2* reg_cut * (current_cut)

        #update

        bu[current_users] = (current_bu - (lr_bu*dldbu)).reshape(list_lens,)
        au[current_users] = (au[current_users]-(lr_au*dldau)).reshape(list_lens,)
        but[current_users, current_days] = (current_but- (lr_but*dldbut)).reshape(list_lens,)
        bi[current_movies] =(current_bi- (lr_bi*dldbi)).reshape(list_lens,)
        bibin[current_movies, current_bins] = (current_bibin -(lr_bibin*dldbibin)).reshape(list_lens,)
        cu[current_users] = (current_cu -(lr_cu* dldcu)).reshape(list_lens,)
        cut[current_users, current_days] =(current_cut - (lr_cut*dldcut)).reshape(list_lens,)
    bar.finish()


    print 'incorporating learned deviations'
    batch_size = 100000



    num_rats = len(probe_I)
    probe_V_test = np.zeros(num_rats)
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
        probe_V_test[i:min(i+batch_size, num_rats)] = (probe_V[i:min(i+batch_size, num_rats)]- pred).reshape(list_lens,)

    prb = math.sqrt(np.sum(np.power(probe_V_test, 2))/num_rats)
    print 'probe rmse base', prb



num_rats = len(I)
V_mod = np.zeros(num_rats)
for i in range(0, num_rats, batch_size):
    current_users = I[i:min(i+batch_size, rat_num)]
    list_lens = len(current_users)
    current_movies = J[i:min(i+batch_size, rat_num)]
    current_days = d[i:min(i+batch_size, rat_num)]
    current_bins = np.floor(current_days/(ndays/nbins))
    current_bu = gpu.garray(bu[current_users])
    current_au = gpu.garray(au[current_users])
    current_devut = gpu.garray(devut[:, i:min(i+batch_size, rat_num)])
    current_but = gpu.garray(but[current_users, current_days].toarray())
    current_bi = gpu.garray(bi[current_movies])
    current_bibin = gpu.garray(bibin[current_movies, current_bins].toarray())
    current_cu = gpu.garray(cu[current_users])
    current_cut = gpu.garray(cut[current_users, current_days].toarray())

    pred = -(- current_bu- current_but-current_au*current_devut- (current_bi+ current_bibin)*(current_cu + current_cut))
    #print pred[0], V[i:min(i+batch_size, num_rats)]
    V_mod[i:min(i+batch_size, num_rats)] = (V[i:min(i+batch_size, num_rats)]- pred).reshape(list_lens,)



but = but.tocoo()
cut = cut.tocoo()
bibin = bibin.tocoo()
print "starting writes"
f = h5py.File(dest_filename, 'w')
f.create_dataset('train_user_list', data = I)
f.create_dataset('train_item_list', data = J)
f.create_dataset('train_rating_list', data = V_mod)
f.create_dataset('train_time_list', data = d)
f.create_dataset('bu', data = bu)
f.create_dataset('au', data = au)
f.create_dataset('butd', data = but.data)
f.create_dataset('butr', data = but.row)
f.create_dataset('butc', data = but.col)
f.create_dataset('bi', data = bi)
f.create_dataset('bibind', data = bibin.data)
f.create_dataset('bibinr', data = bibin.row)
f.create_dataset('bibinc', data = bibin.col)
f.create_dataset('cu', data = cu)
f.create_dataset('cutd', data = cut.data)
f.create_dataset('cutr', data = cut.row)
f.create_dataset('cutc', data = cut.col)
f.create_dataset('probe_user_list', data = probe_I)
f.create_dataset('probe_item_list', data = probe_J)
f.create_dataset('probe_rating_list', data = probe_V_test)
f.create_dataset('probe_time_list', data = probe_d)
f.create_dataset('qual_user_list', data = qual_I)
f.create_dataset('qual_item_list', data = qual_J)
f.create_dataset('qual_time_list', data = qual_d)


print 'finished single arrays'

f.close()

but = but.tolil()
cut = cut.tolil()
bibin = bibin.tolil()

from sklearn.cluster import KMeans

from scipy.sparse import csr_matrix, vstack
from sklearn.linear_model import SGDRegressor
import progressbar
import numpy as np
import h5py
import math
import random




'''

def euclidean_distance_of_users_based_on_ratings(movie_ratings1, movie_ratings2):

    movie_ratings2 = vstack(np.repeat(movie_ratings2, movie_ratings1.shape[0], axis=0))

    return np.sqrt((((movie_ratings1 != 0).multiply(movie_ratings2 != 0).multiply(movie_ratings1 - movie_ratings2)).power(2)).sum(axis = 1))


'''
h5filename = "../um/all.h5"

h5file = h5py.File(h5filename)

users = h5file['train_user_list']
movies = h5file['train_item_list']
ratings = h5file['train_rating_list']

probe_users = h5file['probe_user_list']
probe_movies = h5file['probe_item_list']
probe_ratings = h5file['probe_rating_list']

qual_users = h5file['qual_user_list']
qual_items = h5file['qual_item_list']
print 'loaded data'
nusers = 458293
nitems = 17770
R_m = csr_matrix((ratings,(movies,users)), shape=(nitems, nusers))

R_u = csr_matrix((ratings,(users,movies)), shape=(nusers, nitems))

print 'created mat'
all_movie_stdev = np.std(movies)
all_movie_avg = np.mean(movies)
probe_num = len(probe_ratings)
batch_size = 10000
probe_se = 0

#calculate user offsets

'''user_avgs = R_u.sum(axis=1)/R_u.getnnz(axis=1)
print 'done avg'
user_rating_offsets = user_avgs - ((R_u != 0).multiply(all_movie_avg))
print 'done offset'''
#batched average
movie_avg = np.zeros(nitems)
movie_std = np.zeros(nitems)
for i in range(0, nitems, batch_size):
    if i % 10 == 0:
        print i
    movie_current = R_m[i:min(i+batch_size, nitems), :]
    batch_avg = ((movie_current.sum(axis =1).flatten())/movie_current.getnnz(axis =1))
    movie_avg[i:min(i+batch_size, nitems)] = batch_avg
    movie_std[i:min(i+batch_size, nitems)] = np.sqrt(abs(movie_current.power(2).sum(axis = 1).flatten()/movie_current.getnnz(axis =1) - batch_avg))

user_avg = np.zeros(nusers)
user_std = np.zeros(nusers)
for i in range(0, nusers, batch_size):
    users_current = R_u[i:min(i+batch_size, nusers), :]
    batch_avg = ((users_current.sum(axis =1).flatten())/users_current.getnnz(axis =1))
    user_avg[i:min(i+batch_size, nusers)] = batch_avg
    user_std[i:min(i+batch_size, nusers)] = np.sqrt(abs(users_current.power(2).sum(axis = 1).flatten()/users_current.getnnz(axis =1) - batch_avg))
print 'done avging', movie_avg, user_avg

# sgd fitter
lin_model = SGDRegressor()

rat_num = len(probe_ratings)

for i in range(0, rat_num, batch_size):

    given = probe_ratings[i:min(i+batch_size, rat_num)]
    u_mean = user_avg[probe_users[i:min(i+batch_size, probe_num)]]
    m_mean = movie_avg[probe_movies[i:min(i+batch_size, probe_num)]]
    u_mean = np.array([u_mean]).T
    m_mean = np.array([m_mean]).T

    preding = np.concatenate((u_mean, m_mean), axis = 1)

    lin_model.partial_fit(preding, given)



#prediction
for i in range(0, probe_num, batch_size):


    u_mean = user_avg[probe_users[i:min(i+batch_size, probe_num)]]
    #print u_mean
    #print u_mean, 'u_mean', users_current.getnnz(axis =1), 'uzx'
    '''
    u_stdev = np.sqrt(((np.sum(np.power((users_current - u_mean*(users_current!=0)),2), axis=1).flatten())/users_current.getnnz(axis=1))).T
    print u_stdev
    if u_stdev <0.2 and users_current.getnnz() > 5:
        pred = users_current.sum(axis=1)/users_current.getnnz(axis=1)
        print pred, 'pred last'
    else:
        movie_current = R_m[probe_movies[i:min(i+batch_size, probe_num)], :]
        m_mean = movie_current.sum(axis=1)/movie_current.getnnz(axis=1)
        print movie_current.getnnz(axis=1), 'sd'
        m_stdev = np.sqrt((np.sum(np.power((movie_current - m_mean*(movie_current!=0)),2),axis=1).flatten())/movie_current.getnnz(axis=1)).T
        pred = movie_current.sum(axis=1)/movie_current.getnnz(axis=1)+(u_mean-all_movie_avg)*(m_stdev/all_movie_stdev)
        print pred, 'pred last'
        '''

    m_mean = movie_avg[probe_movies[i:min(i+batch_size, probe_num)]]

    pred = 0.5*u_mean + 0.5*m_mean
    #print u_mean, m_mean, pred
    given = probe_ratings[i:min(i+batch_size, probe_num)]



    pred = np.maximum(np.minimum(pred, (5-3-0.60951619727280626)*np.ones(len(pred))),(1-3-0.60951619727280626)*np.ones(len(pred)))


    probe_se += np.sum(np.power((given-pred),2))

    #print math.sqrt(probe_se/(i+batch_size))



print 'naive avg', math.sqrt(probe_se/probe_num)

probe_se = 0
for i in range(0, probe_num, batch_size):


    u_mean = user_avg[probe_users[i:min(i+batch_size, probe_num)]]
    #print u_mean
    #print u_mean, 'u_mean', users_current.getnnz(axis =1), 'uzx'
    '''
    u_stdev = np.sqrt(((np.sum(np.power((users_current - u_mean*(users_current!=0)),2), axis=1).flatten())/users_current.getnnz(axis=1))).T
    print u_stdev
    if u_stdev <0.2 and users_current.getnnz() > 5:
        pred = users_current.sum(axis=1)/users_current.getnnz(axis=1)
        print pred, 'pred last'
    else:
        movie_current = R_m[probe_movies[i:min(i+batch_size, probe_num)], :]
        m_mean = movie_current.sum(axis=1)/movie_current.getnnz(axis=1)
        print movie_current.getnnz(axis=1), 'sd'
        m_stdev = np.sqrt((np.sum(np.power((movie_current - m_mean*(movie_current!=0)),2),axis=1).flatten())/movie_current.getnnz(axis=1)).T
        pred = movie_current.sum(axis=1)/movie_current.getnnz(axis=1)+(u_mean-all_movie_avg)*(m_stdev/all_movie_stdev)
        print pred, 'pred last'
        '''

    m_mean = movie_avg[probe_movies[i:min(i+batch_size, probe_num)]]
    u_mean = np.array([u_mean]).T
    m_mean = np.array([m_mean]).T
    preding = np.concatenate((u_mean, m_mean), axis = 1)
    pred = lin_model.predict(preding)
    #print u_mean, m_mean, pred
    given = probe_ratings[i:min(i+batch_size, probe_num)]



    pred = np.maximum(np.minimum(pred, (5-3-0.60951619727280626)*np.ones(len(pred))),(1-3-0.60951619727280626)*np.ones(len(pred)))


    probe_se += np.sum(np.power((given-pred),2))

    #print math.sqrt(probe_se/(i+batch_size))

print 'trained avg1', math.sqrt(probe_se/probe_num)

# sgd fitter
lin_model = SGDRegressor()

rat_num = len(probe_ratings)

for i in range(0, rat_num, batch_size):

    given = probe_ratings[i:min(i+batch_size, rat_num)]
    u_mean = user_avg[probe_users[i:min(i+batch_size, probe_num)]]
    m_mean = movie_avg[probe_movies[i:min(i+batch_size, probe_num)]]
    u_std = user_std[probe_users[i:min(i+batch_size, probe_num)]]
    m_std = movie_std[probe_movies[i:min(i+batch_size, probe_num)]]

    u_mean = np.array([u_mean]).T
    m_mean = np.array([m_mean]).T
    u_std = np.array([u_std]).T
    m_std = np.array([m_std]).T

    preding = np.concatenate((u_mean, u_std, m_mean, m_std), axis = 1)

    lin_model.partial_fit(preding, given)

    if (i % 1000000) == 0:
        print lin_model.score(preding, given)

probe_se = 0
for i in range(0, probe_num, batch_size):


    u_mean = user_avg[probe_users[i:min(i+batch_size, probe_num)]]
    u_std = user_std[probe_users[i:min(i+batch_size, probe_num)]]
    m_std = movie_std[probe_movies[i:min(i+batch_size, probe_num)]]
    #print u_mean
    #print u_mean, 'u_mean', users_current.getnnz(axis =1), 'uzx'
    '''
    u_stdev = np.sqrt(((np.sum(np.power((users_current - u_mean*(users_current!=0)),2), axis=1).flatten())/users_current.getnnz(axis=1))).T
    print u_stdev
    if u_stdev <0.2 and users_current.getnnz() > 5:
        pred = users_current.sum(axis=1)/users_current.getnnz(axis=1)
        print pred, 'pred last'
    else:
        movie_current = R_m[probe_movies[i:min(i+batch_size, probe_num)], :]
        m_mean = movie_current.sum(axis=1)/movie_current.getnnz(axis=1)
        print movie_current.getnnz(axis=1), 'sd'
        m_stdev = np.sqrt((np.sum(np.power((movie_current - m_mean*(movie_current!=0)),2),axis=1).flatten())/movie_current.getnnz(axis=1)).T
        pred = movie_current.sum(axis=1)/movie_current.getnnz(axis=1)+(u_mean-all_movie_avg)*(m_stdev/all_movie_stdev)
        print pred, 'pred last'
        '''

    m_mean = movie_avg[probe_movies[i:min(i+batch_size, probe_num)]]
    u_mean = np.array([u_mean]).T
    m_mean = np.array([m_mean]).T
    u_std = np.array([u_std]).T
    m_std = np.array([m_std]).T
    preding = np.concatenate((u_mean, u_std, m_mean, m_std), axis = 1)
    pred = lin_model.predict(preding)
    #print u_mean, m_mean, pred
    given = probe_ratings[i:min(i+batch_size, probe_num)]



    pred = np.maximum(np.minimum(pred, (5-3-0.60951619727280626)*np.ones(len(pred))),(1-3-0.60951619727280626)*np.ones(len(pred)))


    probe_se += np.sum(np.power((given-pred),2))

    #print math.sqrt(probe_se/(i+batch_size))

print 'trained avg2', math.sqrt(probe_se/probe_num)
pred_file = open('predictions.txt', 'w')

rat_num = len(qual_items)

for i in range(0, rat_num, batch_size):
    u_mean = user_avg[qual_users[i:min(i+batch_size, rat_num)]]
    m_mean = movie_avg[qual_items[i:min(i+batch_size, rat_num)]]
    u_mean = np.array([u_mean]).T
    m_mean = np.array([m_mean]).T
    preding = np.concatenate((u_mean, m_mean), axis = 1)
    pred = lin_model.predict(preding)
    #pred = np.maximum(np.minimum(pred, (5-3-0.60951619727280626)*np.ones(len(pred))),(1-3-0.60951619727280626)*np.ones(len(pred)))
    for pr in pred:
        pred_file.write('%.3f\n' % max(min((pr+3+0.60951619727280626),5),1)) #convert back to 1-5 ratings systems


print probe_se
print probe_num
print math.sqrt(probe_se/probe_num)

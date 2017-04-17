# Module for importing data into scipy sparse arrays, with progressbar
# Aadith Moorthy

import scipy.sparse
import envoy
import progressbar
import os.path

nusers = 7#458293
nitems = 17770
# Returns scipy.sparse matrices, for training and probe
# optimized for getting individual users data
# users are rows, movies are columns



# Returns scipy.sparse matrices, for training and probe
# optimized for getting individual movies data
# movies are rows, users are columns
def import_data_movie(trainpath, probepath):
    assert(os.path.isfile(trainpath))

    assert(os.path.isfile(probepath))

    Dtr = get_train_data_movie(trainpath)

    Dpr = get_probe_data_movie(probepath)

    return Dtr.tocsr(), Dpr.tocsr()




# Returns scipy.sparse matrices, for training and probe
# optimized for getting individual users data
# users are rows, movies are columns
def import_data_user(trainpath, probepath):
    assert(os.path.isfile(trainpath))

    assert(os.path.isfile(probepath))

    Dtr = get_train_data_user(trainpath)

    Dpr = get_probe_data_user(probepath)

    return Dtr.tocsr(), Dpr.tocsr()

###############################################################################
# Helper methods
def get_train_data_user(filename):



    r = envoy.run('wc -l {}'.format(filename))
    num_lines = int(r.std_out.strip().partition(' ')[0])
    bar = progressbar.ProgressBar(maxval=num_lines, widgets=["Loading train ratings: ",
                                                             progressbar.Bar(
                                                                 '=', '[', ']'),
                                                             ' ', progressbar.Percentage(),

                                                             ' ', progressbar.ETA()]).start()
    I, J, V = [], [], []


    with open(filename) as f:
        for i, line in enumerate(f):

            if (i % 1000) == 0:
                bar.update(i % bar.max_value)
            userid, itemid, date, rating = line.split()



            I.append(int(userid)-1)
            J.append(int(itemid)-1)
            V.append(float(rating)-3) #Center around 3, by setting 3 to 0

    bar.finish()

    R = scipy.sparse.coo_matrix(
        (V, (I, J)), shape=(nusers, nitems))



    return R


def get_train_data_movie(filename):



    r = envoy.run('wc -l {}'.format(filename))
    num_lines = int(r.std_out.strip().partition(' ')[0])
    bar = progressbar.ProgressBar(maxval=num_lines, widgets=["Loading train ratings: ",
                                                             progressbar.Bar(
                                                                 '=', '[', ']'),
                                                             ' ', progressbar.Percentage(),

                                                             ' ', progressbar.ETA()]).start()
    I, J, V = [], [], []



    with open(filename) as f:
        for i, line in enumerate(f):

            if (i % 1000) == 0:
                bar.update(i % bar.max_value)
            userid, itemid, date, rating = line.split()



            I.append(int(userid)-1)
            J.append(int(itemid)-1)
            V.append(float(rating)-3) #Center around 3, by setting 3 to 0

    bar.finish()

    R = scipy.sparse.coo_matrix(
        (V, (J, I)), shape=(nitems, nusers))



    return R


def get_probe_data_user(testpath):
    r = envoy.run('wc -l {}'.format(testpath))
    num_lines = int(r.std_out.strip().partition(' ')[0])
    bar = progressbar.ProgressBar(maxval=num_lines, widgets=['Loading test ratings: ',
                                                             progressbar.Bar(
                                                                 '=', '[', ']'),
                                                             ' ', progressbar.Percentage(),

                                                             ' ', progressbar.ETA()]).start()



    I, J, V = [], [], []

    with open(testpath) as fp:
        for i, line in enumerate(fp):
            if (i % 1000) == 0:
                bar.update(i % bar.max_value)
            user, item, date, rating = line.split()


            I.append(int(user)-1)
            J.append(int(item)-1)
            V.append(float(rating)-3) #Center around 3 by setting 3 to 0



    bar.finish()
    R = scipy.sparse.coo_matrix(
        (V, (I, J)), shape=(nusers, nitems))



    return R


def get_probe_data_movie(testpath):
    r = envoy.run('wc -l {}'.format(testpath))
    num_lines = int(r.std_out.strip().partition(' ')[0])
    bar = progressbar.ProgressBar(maxval=num_lines, widgets=['Loading test ratings: ',
                                                             progressbar.Bar(
                                                                 '=', '[', ']'),
                                                             ' ', progressbar.Percentage(),

                                                             ' ', progressbar.ETA()]).start()



    I, J, V = [], [], []

    with open(testpath) as fp:
        for i, line in enumerate(fp):
            if (i % 1000) == 0:
                bar.update(i % bar.maxval)
            user, item, date, rating = line.split()


            I.append(int(user)-1)
            J.append(int(item)-1)
            V.append(float(rating)-3) #Center around 3 by setting 3 to 0

    bar.finish()
    R = scipy.sparse.coo_matrix(
        (V, (J, I)), shape=(nitems, nusers))

    return R



#if __name__ == '__main__':
    #Dtr, Dpr = import_data_movie("../um/small_train.dta", "../um/small_train.dta")


   # print len(Dtr.indices)

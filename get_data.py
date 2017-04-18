# Module for importing data into scipy sparse arrays, with progressbar
# Aadith Moorthy

import scipy.sparse
import envoy
import progressbar
import os.path

nusers = 458293
nitems = 17770



# Returns scipy.sparse matrices, for training and probe
# optimized for getting individual movies data
# movies are rows, users are columns
# beware, output ratings are shifted to -2,-1,0,1,2
def import_data_movie(trainpath, probepath):
    assert(os.path.isfile(trainpath))

    assert(os.path.isfile(probepath))

    Dtr = get_train_data_movie(trainpath)

    Dpr = get_probe_data_movie(probepath)

    print "finished compiling ratings"
    Dtr = Dtr.tocsr()
    Dpr = Dpr.tocsr()
    return Dtr, Dpr




# Returns scipy.sparse matrices, for training and probe
# optimized for getting individual users data
# users are rows, movies are columns
# beware, output ratings are shifted to -2,-1,0,1,2
def import_data_user(trainpath, probepath):
    assert(os.path.isfile(trainpath))

    assert(os.path.isfile(probepath))

    Dtr = get_train_data_user(trainpath)

    Dpr = get_probe_data_user(probepath)

    print "finished compiling ratings"
    Dtr = Dtr.tocsr()
    Dpr = Dpr.tocsr()
    return Dtr, Dpr

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

    # to remove later
    if 'all.dta' in filename:
        idx_file = open(filename[:-3]+'idx')
        global probe_data_preload_I
        global probe_data_preload_J
        global probe_data_preload_V
        probe_data_preload_I = []
        probe_data_preload_J = []
        probe_data_preload_V = []


    with open(filename) as f:
        for i, line in enumerate(f):
            if (i % 1000) == 0:
                bar.update(i % bar.maxval)

            userid, itemid, date, rating = line.split()

            if 'all.dta' in filename:
                idx = int(idx_file.readline().replace("\n",""))
                if idx <= 3:

                    I.append(int(userid)-1)
                    J.append(int(itemid)-1)
                    V.append(float(rating)-3)
                elif idx == 4:
                    probe_data_preload_I.append(int(userid)-1)
                    probe_data_preload_J.append(int(itemid)-1)
                    probe_data_preload_V.append(float(rating)-3)

            else:

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
    I = []
    J = []
    V = []

    # to remove later
    if 'all.dta' in filename:
        idx_file = open(filename[:-3]+'idx')
        global probe_data_preload_I
        global probe_data_preload_J
        global probe_data_preload_V
        probe_data_preload_I = []
        probe_data_preload_J = []
        probe_data_preload_V = []


    with open(filename) as f:
        for i, line in enumerate(f):
            if (i % 1000) == 0:
                bar.update(i % bar.maxval)

            userid, itemid, date, rating = line.split()

            if 'all.dta' in filename:
                idx = int(idx_file.readline().replace("\n",""))
                if idx <= 3:

                    I.append(int(userid)-1)
                    J.append(int(itemid)-1)
                    V.append(float(rating)-3)
                elif idx == 4:
                    probe_data_preload_I.append(int(userid)-1)
                    probe_data_preload_J.append(int(itemid)-1)
                    probe_data_preload_V.append(float(rating)-3)

            else:

                I.append(int(userid)-1)
                J.append(int(itemid)-1)
                V.append(float(rating)-3)
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

    if 'probe_data_preload_I' in globals():
        I = probe_data_preload_I
        J = probe_data_preload_J
        V = probe_data_preload_V
    else:
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

    if 'probe_data_preload_I' in globals():
        I = probe_data_preload_I
        J = probe_data_preload_J
        V = probe_data_preload_V
    else:
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



if __name__ == '__main__':
    Dtr, Dpr = import_data_movie("../um/small_train.dta", "../um/small_train.dta")

    for i in Dtr:
        if i != None:
            print i

    print len(Dtr.indices)

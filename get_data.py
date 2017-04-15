# Module for importing data into scipy sparse arrays, with progressbar
# Aadith Moorthy

import scipy.sparse
import envoy
import progressbar
import os.path

# Returns 2 scipy.sparse matrices, for training and probe
def import_data(trainpath, probepath):
    assert(os.path.isfile(trainpath))


    assert(os.path.isfile(probepath))

    Dtr, users, items = get_train_data(trainpath)

    Dpr = get_probe_data(probepath,  users, items)

    return Dtr, Dpr


###############################################################################
# Helper methods

def get_train_data(filename):
    users = {}
    items = {}
    nusers = 0
    nitems = 0
    include_time = False
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
                bar.update(i % bar.maxval)
            userid, itemid, date, rating = line.split()
            if userid not in users:
                users[userid] = nusers
                nusers += 1
            if itemid not in items:
                items[itemid] = nitems
                nitems += 1
            uid = users[userid]
            iid = items[itemid]
            I.append(uid)
            J.append(iid)
            V.append(float(rating))
    bar.finish()

    R = scipy.sparse.coo_matrix(
        (V, (I, J)), shape=(nusers, nitems))
    R = R.tocsr()
    return R, users, items


def get_probe_data(testpath, users, items):
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
            if user in users and item in items:
                I.append(users[user])
                J.append(items[item])
                V.append(float(rating))

    bar.finish()
    R = scipy.sparse.coo_matrix(
        (V, (I, J)), shape=(len(users.keys()), len(items.keys())))
    return R.tocsr()


if __name__ == '__main__':
    Dtr, Dpr = import_data("../mu/all.dta", "../mu/small_train.dta")

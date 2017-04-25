import numpy as np
import pandas as pd
import scipy.sparse
import random
import math as m
import get_data_progbar2 as gd

trainPath = "um/small_train.dta"
probePath = 'probe.dta'

M = 458293 						 # Number of users
N = 17770						 # Number of movies


class KMeans:
    def __init__(self, filename):
        self.filename = filename
        self.numUsers = 0
        self.numMovies = 0
        self.userMeans = []
        self.userStds = []
        self.trainingMatrixCoords = None
        self.trainingMatrix = None

    def loadData(self):
        I, J, V = [], [], []
        setUsers = set()
        setMovies = set()
        with open(self.filename) as f:
            for i, line in enumerate(f):
                userid, itemid, date, rating = line.split()
                I.append(int(userid) - 1)
                J.append(int(itemid) - 1)
                V.append(float(rating))

                setUsers.add(int(userid) - 1)
                setMovies.add(int(itemid) - 1)
        self.numUsers = max(setUsers) + 1
        self.numMovies = max(setMovies) + 1
        self.trainingMatrixCoords = scipy.sparse.coo_matrix((V, (I, J)),
                                                            shape=(max(setUsers) + 1, max(setMovies) + 1))
        self.trainingMatrix = self.trainingMatrixCoords.tocsr().toarray()

    def normalizeData(self):
        userSpecific = dict()
        for user, movie, rating in zip(self.trainingMatrixCoords.row, self.trainingMatrixCoords.col, self.trainingMatrixCoords.data):
            if user not in userSpecific:
                userSpecific[user] = [rating]
            else:
                userSpecific[user].append(rating)
        for key in userSpecific:
            print(key)
        return


def main():
    testingData = KMeans(trainPath)
    testingData.loadData()
    testingData.normalizeData()


if __name__ == '__main__':
    main()

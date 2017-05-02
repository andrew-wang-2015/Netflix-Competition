from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os
import pickle

from surprise import BaselineOnly
from surprise import Dataset
from surprise import evaluate
from surprise import Reader
from surprise import SVD
from surprise import accuracy
from surprise import dump
#train.dta and probe.dta in format user, movie, date, rating  

train_file = os.path.expanduser("train.dta")
test_file = os.path.expanduser("probe.dta")
reader = Reader(line_format='user item timestamp rating', sep=' ')

folds_files = [(train_file, test_file)]
data = Dataset.load_from_folds(folds_files, reader=reader)
print("Loaded in training and testing \n")

bsl_options = {'method': 'sgd',
               'learning_rate': .00005,
               }
algo = BaselineOnly(bsl_options=bsl_options)

for trainset, testset in data.folds():
    print ("Started training ... \n")
    # train and test algorithm.
    algo.train(trainset)
    predictions = algo.test(testset)
    
    print("Done training...\n")

    # Compute and print Root Mean Squared Error
    rmse = accuracy.rmse(predictions, verbose=True)

print(predictions)

text_file = open("Baseline_SurpriseResults.txt", "w")
text_file.write("RMSE: %s" % rmse)
text_file.close()




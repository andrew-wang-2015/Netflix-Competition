import os
from surprise import BaselineOnly
from surprise import Dataset
from surprise import evaluate
from surprise import Reader
from surprise import SVD
from surprise import accuracy
#train.dta and probe.dta in format user, movie, date, rating  

train_file = os.path.expanduser("train.dta")
test_file = os.path.expanduser("probe.dta")
reader = Reader(line_format='user item timestamp rating', sep=' ')
train_data = Dataset.load_from_file(train_file, reader=reader)
print("Loaded training data")
test_data = Dataset.load_from_file(test_file, reader=reader)
print("Loaded testing data")

algo = SVD()


algo.train(trainset)
predictions = algo.test(testset)
    
rmse = accuracy.rmse(predictions, verbose = True)


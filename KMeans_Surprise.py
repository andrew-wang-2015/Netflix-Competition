import os
from surprise import BaselineOnly
from surprise import Dataset
from surprise import evaluate
from surprise import Reader
from surprise import SVD
from surprise import accuracy
from surprise import KNNBasic
#train.dta and probe.dta in format user, movie, date, rating   

train_file = os.path.expanduser("train.dta")
test_file = os.path.expanduser("probe.dta")
reader = Reader(line_format='user item timestamp rating', sep=' ')

folds_files = [(train_file, test_file)]
data = Dataset.load_from_folds(folds_files, reader=reader)
print("Loaded in training and testing \n")


sim_options = {'name': 'pearson_baseline',
               'user_based': True  # compute  similarities between users
               }

algo = KNNBasic(sim_options=sim_options)

for trainset, testset in data.folds():
    print ("Started training ... \n")
    # train and test algorithm.
    algo.train(trainset)
    predictions = algo.test(testset)
    
    print("Done training...\n")

    # Compute and print Root Mean Squared Error
    rmse = accuracy.rmse(predictions, verbose=True)

print(predictions)

text_file = open("KMeans_SurpriseResults.txt", "w")
text_file.write("RMSE: %s" % rmse)
text_file.close()




import numpy as np
import pickle
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics.pairwise import cosine_similarity
import io

def calculateRMSE(y_predicted, y_actual):
    '''
    Returns rms between two lists
    '''
    rms = sqrt(mean_squared_error(y_actual, y_predicted))
    return rms
        


def getRating(movieID, user, validNeighbors, userMovieDict, userRatings, movieRatings, moviesRated):
    '''
    Given neighbors to search, we actually generate a rating 
    '''
    
    # We return average of averages if there are no valid neighbors
    if len(validNeighbors) == 0:
        return (sum(userRatings[user])/ len(userRatings[user]) + sum(movieRatings[movieID])/ len(movieRatings[movieID])) / 2
        
    
    numerator = 0
    denominator = 0
    for neighbor in validNeighbors:
        intersection = userMovieDict[user] & userMovieDict[neighbor]
        userRatings = []
        neighborRatings = []
        for m in intersection:
            userRatings.append(moviesRated[user][m])
            neighborRatings.append(moviesRated[neighbor][m])
        corr = (stats.pearsonr(userRatings, neighborRatings)[0] + 1) / 2   # To scale to [0,1]... maybe use cosine later
        numerator += corr * moviesRated[neighbor][m]
        denominator += corr
    del intersection
    del userRatings
    del neighborRatings
    del validNeighbors
    return (numerator / denominator)

def pruneValidNeighbors(movieID, user, validNeighbors, userMovieDict):
    
    '''
    This takes the original neighbors set and further prunes it by filtering out
    those neighbros which haven't rated the movie of choice
    '''
    realValidNeighbors = set()
    for i in validNeighbors[user]:
        if movieID in userMovieDict[i]:
            realValidNeighbors.add(i)
    return realValidNeighbors

def main():
    #df = pd.read_table("small_train.dta", sep=" ", names = ["User", "Movie", "Date", "Rating"], index_col = False)
    R = []
    U = []
    M = []

    with open('train.dta','r') as f:
        for line in f:
            u, m, d, r = line.split()
            R.append(r)
            U.append(u)
            M.append(m)        
    f.close()


    Ratings = np.array(R, dtype = np.uint8)
    Users = np.array(U, dtype =np.uint16)
    Movies = np.array(M, dtype = np.uint16)
    del R
    del U
    del M

    print("Loaded all data")
    numUsers = 7
    moviesRated = dict()   # Keys are users, value for each key is dictionary mapping movie to rating
    rated = dict()
    onlyUserMovie = dict()    # Keys are users, value for each key is set of movies that user has rated
    validNeighbors = dict()
    userRatings = dict()
    movieRatings = dict()
    
    
    for index, user in enumerate(Users):
        
        # Make a dict that maps users to their movie ratings
        if user not in userRatings:
            userRatings[user] = [Ratings[index]]
        else:
            userRatings[user].append(Ratings[index])
        
        # Make a dict that maps movies to their movie ratings
        if Movies[index] not in movieRatings:
            movieRatings[Movies[index]] =[ Ratings[index]]
        else:  
            movieRatings[Movies[index]].append(Ratings[index])
            
        
        if index > 0 and Users[index] != Users[index-1]:
            moviesRated[Users[index-1]] = rated
            rated = dict()
            
        rated[Movies[index]] = Ratings[index]
        
        # Key: user, value: movie numbers
        if user in onlyUserMovie:
            onlyUserMovie[user].add(Movies[index])
        else:
            onlyUserMovie[user] = {Movies[index]}
    # To get the last user in the moviesRated dict   
    moviesRated[Users[len(Users) - 1]] = rated    
    
    with open('MoviesRated.pickle', 'wb') as handle:
        pickle.dump(moviesRated, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    with open('OnlyUserMovie.pickle', 'wb') as handle:
        pickle.dump(onlyUserMovie, handle, protocol=pickle.HIGHEST_PROTOCOL)    
        
    with open('UserRatings.pickle', 'wb') as handle:
        pickle.dump(userRatings, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    with open('MovieRatings.pickle', 'wb') as handle:
        pickle.dump(movieRatings, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    for i in range(1,numUsers + 1):
        validNeighbors[i] = set()
    

    
    # For each user, we have a dictionary of "valid neighbors," as designated by sharing 25% movies in common having been rated
    for i in range(1,numUsers):   # Our UserIDs range from 1 to 7
        for j in range(i+1, numUsers + 1):
            if len(onlyUserMovie[i] & onlyUserMovie[j]) >= 0.25 * min(len(onlyUserMovie[i]), len(onlyUserMovie[j])):
                validNeighbors[i].add(j)
                validNeighbors[j].add(i)    
    with open('ValidNeighbors.pickle', 'wb') as handle:
        pickle.dump(validNeighbors, handle, protocol=pickle.HIGHEST_PROTOCOL) 
        
    predictions = []
    for index, user in enumerate(Users):
        realNeighbors = pruneValidNeighbors(Movies[index], user, validNeighbors, onlyUserMovie)
        predictions.append(getRating(Movies[index], user, realNeighbors, onlyUserMovie, userRatings, movieRatings, moviesRated))    
        
    print(calculateRMSE(predictions, Ratings))
main()
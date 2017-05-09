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
        


def getRating(movieID, user, validNeighbors, userMovieDict, moviesRated):
    '''
    Given neighbors to search, we actually generate a rating 
    '''
    
    # We return average of averages if there are no valid neighbors
    if len(validNeighbors) == 0:
        return (3.6)
        
    
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
    numUsers = 458293 
    moviesRated = dict()   # Keys are users, value for each key is dictionary mapping movie to rating
    rated = dict()
    onlyUserMovie = dict()    # Keys are users, value for each key is set of movies that user has rated
    validNeighbors = dict()   
    Users = []
    Movies = []
    Ratings = []
    prevUser = 1
    with open('train.dta','r') as f:
        for line in f:
            user, movie, date, rating = line.split()
            user = np.uint32(user)
            if user % 10000 == 0 and user != prevUser:
                print(user)
            movie = np.uint16(movie)
            rating = np.uint8(rating)
            if user not in onlyUserMovie:
                onlyUserMovie[user] = {movie}
            else:
                onlyUserMovie[user].add(movie)     
            
            #if user != prevUser:
                #moviesRated[prevUser] = rated 
                #prevUser = user 
                #rated = dict()
                
            #rated[movie] = rating 
    #moviesRated[prevUser] = rated 
            
            
                
    f.close()
    #print(onlyUserMovie)

    print("Loaded all data")
 
    
    #with open('MoviesRated.pickle', 'wb') as handle:
        #pickle.dump(moviesRated, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    #with open('OnlyUserMovie.pickle', 'wb') as handle:
        #pickle.dump(onlyUserMovie, handle, protocol=pickle.HIGHEST_PROTOCOL)    
        
    for i in range(1,numUsers + 1):
        validNeighbors[i] = set()
    
    # For each user, we have a dictionary of "valid neighbors," as designated by sharing 25% movies in common having been rated
    for i in range(1,numUsers):   # Our UserIDs range from 1 to 7
        for j in range(i+1, numUsers + 1):
            if len(onlyUserMovie[i] & onlyUserMovie[j]) >= 0.25 * min(len(onlyUserMovie[i]), len(onlyUserMovie[j])):
                validNeighbors[i].add(j)
                validNeighbors[j].add(i)    
    noNeighbors = 0
    for i in validNeighbors:
        if len(validNeighbors[i]) == 0:
            noNeighbors += 1
    print(noNeighbors)
            
    #with open('ValidNeighbors.pickle', 'wb') as handle:
        #pickle.dump(validNeighbors, handle, protocol=pickle.HIGHEST_PROTOCOL) 
    print("Generated neighbors for all users")
    Ratings = []
    predictions = []
    # Load in test file
    with open('probe.dta','r') as f:
        for line in f:
            user, movie, date, rating = line.split()
            user = np.uint32(user)
            movie = np.uint16(movie)
            rating = np.uint8(rating)
            Ratings.append(rating)
            realNeighbors = pruneValidNeighbors(movie, user, validNeighbors, onlyUserMovie)
            predictions.append(getRating(movie, user, realNeighbors, onlyUserMovie, moviesRated))   
            
    print(calculateRMSE(predictions, Ratings))
    
main()
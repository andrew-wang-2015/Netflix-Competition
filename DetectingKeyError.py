import numpy as np
import pandas as pd
import gc
from scipy import stats
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics.pairwise import cosine_similarity

def main():
    gc.enable()
    numUsers = 458293 
    numMovies = 17770
    moviesRated = dict()   # Keys are users, value for each key is dictionary mapping movie to rating
    rated = dict()
    onlyUserMovie = dict()    # Keys are users, value for each key is set of movies that user has rated
    validNeighbors = dict()
    movieRatings = dict() 
    movieUser = dict()
    prevUser = 1
    onlyUserMovie = dict()
    # total queries in probe.dta is 1374739
    trainingFile = 'Train_Without_Dates.dta'
    testingFile = 'Probe_Without_Dates.dta'
    with open(testingFile,'r') as f:
        for line in f:            
            user, movie, rating = line.split()
            movie = np.uint16(movie)
            movieUser[movie] = set()
    f.close()
    
    gc.collect()
            
    for i in range(1, numUsers + 1):
        onlyUserMovie[i] = set()
    with open(trainingFile,'r') as f:
        for line in f:
            user, movie, rating = line.split()
            user = np.uint32(user)
            if user % 10000 == 0 and user != prevUser:
                print(user)
            movie = np.uint16(movie)
            rating = np.uint8(rating)
            onlyUserMovie[user].add(movie)
            if movie in movieUser:
                movieUser[movie].add(user)
            
            if user != prevUser:
                moviesRated[prevUser] = rated 
                prevUser = user 
                rated = dict()
                
            rated[movie] = rating 
    moviesRated[user] = rated 
            
            
                
    f.close()
    gc.collect()

    print("Loaded all data")

    Ratings = []
    predictions = []
    count = 0   
    lineCount = 0
    # Load in test file
    
    with open('testingFile.dta','r') as f:
        
        for line in f:            
            lineCount += 1
            if lineCount % 10000 == 0:
                print(lineCount)
            user, movie, rating = line.split()
            user = np.uint32(user)
            movie = np.uint16(movie)
            rating = np.uint8(rating)
            Ratings.append(rating)

            legitNeighbors = []
            usersThatRated = movieUser[movie]
            for i in usersThatRated:
                if len(onlyUserMovie[i] & onlyUserMovie[user]) >= 0.5 * len(onlyUserMovie[user]):
                    legitNeighbors.append(i)
            numerator = 0
            denominator = 0
            if len(legitNeighbors) == 0:
                count += 1
                predictions.append(3.61)
                continue
            for neighbor in legitNeighbors:
                userRatings = []
                neighborRatings = []  
                for m in (onlyUserMovie[user] & onlyUserMovie[neighbor]):
                    userRatings.append(moviesRated[user][m])
                    neighborRatings.append(moviesRated[neighbor][m])
                corr = (stats.pearsonr(userRatings, neighborRatings)[0] + 1) / 2   # To scale to [0,1]... maybe use cosine later
                if corr > 0.1:
                    numerator += corr * moviesRated[neighbor][m]
                    denominator += corr
            predictions.append(numerator/denominator)
            del legitNeighbors
            del userRatings
            del neighborRatings
            gc.collect()
            
    print("Number of users with no valid neighbors " + str(count))   
    rms = sqrt(mean_squared_error(predictions, Ratings))
            
    print(rms)
    
main()
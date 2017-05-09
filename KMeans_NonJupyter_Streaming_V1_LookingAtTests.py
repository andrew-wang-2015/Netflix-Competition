import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics.pairwise import cosine_similarity

def main():
    numUsers = 458293 
    moviesRated = dict()   # Keys are users, value for each key is dictionary mapping movie to rating
    rated = dict()
    onlyUserMovie = dict()    # Keys are users, value for each key is set of movies that user has rated
    validNeighbors = dict()
    movieRatings = dict() 
    movieUser = dict()
    prevUser = 1
    onlyUserMovie = dict()
    with open('Train_Without_Dates.dta','r') as f:
        for line in f:
            user, movie, rating = line.split()
            user = np.uint32(user)
            if user % 10000 == 0 and user != prevUser:
                print(user)
            movie = np.uint16(movie)
            rating = np.uint8(rating)
            if user not in onlyUserMovie:
                #userRatings[user] = [rating] 
                onlyUserMovie[user] = {movie}
            else:
                #userRatings[user].append(rating)
                onlyUserMovie[user].add(movie)
            if movie not in movieUser:
                movieUser[movie] = {user}
            else:
                movieUser[movie].add(user)
            
            if user != prevUser:
                moviesRated[prevUser] = rated 
                prevUser = user 
                rated = dict()
                
            rated[movie] = rating 
    moviesRated[prevUser] = rated 
            
            
                
    f.close()

    print("Loaded all data")

    Ratings = []
    predictions = []
       
    # Load in test file

    with open('small_train.dta','r') as f:
        count = 0
        for line in f:            
            user, movie, date, rating = line.split()
            user = np.uint32(user)
            movie = np.uint16(movie)
            rating = np.uint8(rating)
            Ratings.append(rating)

            legitNeighbors = []
            usersThatRated = movieUser[movie]
            for i in usersThatRated:
                if len(onlyUserMovie[i] & onlyUserMovie[user]) >= 0.25 * min(len(onlyUserMovie[i]), len(onlyUserMovie[user])):
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
                for m in onlyUserMovie[user] & onlyUserMovie[neighbor]:
                    userRatings.append(moviesRated[user][m])
                    neighborRatings.append(moviesRated[neighbor][m])
                corr = (stats.pearsonr(userRatings, neighborRatings)[0] + 1) / 2   # To scale to [0,1]... maybe use cosine later
                numerator += corr * moviesRated[neighbor][m]
                denominator += corr
            predictions.append(numerator/denominator)
            del legitNeighbors
            del userRatings
            del neighborRatings
            
    print(count)        
    rms = sqrt(mean_squared_error(predictions, Ratings))
            
    print(rms)
    
main()
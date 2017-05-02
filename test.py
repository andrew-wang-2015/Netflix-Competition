import pickle 
with open('MoviesRated.pickle', 'rb') as handle:
    b = pickle.load(handle)
print(b)
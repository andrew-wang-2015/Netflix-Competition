# CS 156 SVD

import numpy as np
import pandas as pd
import scipy.sparse
import random
import math as m
import get_data_progbar2 as gd

M = 458293 						 # Number of users
N = 17770						 # Number of movies
K = 20							 # Number of latent factors
l = 0.1							 # Regularization coefficient
eta = 0.04						 # Step size for SGD
epsilon = 0.00001				 # Threshold for stopping condition
maxIterations = 3000000          # Max iterations

# Adjust path as needed
trainPath = 'train.dta'
probePath = 'probe.dta'

def loss(mat, U, V):
	''' Training objective '''
	ratings = mat.tocoo()
	total = 0
	UT = np.transpose(U)
	for i, j, y in zip(ratings.row, ratings.col, ratings.data):
		predicted = np.dot(UT[i, :], V[:, j])
		total += (predicted - y) ** 2

	return total

def main():
	train = gd.get_train_data_user(trainPath).tocsr()
	errors = []

	# Initialize the U and V matrices
	U = np.random.uniform(low=-0.5, high=0.5, size=(K, M + 1))
	print('Made U')
	V = np.random.uniform(low=-0.5, high=0.5, size=(K, N + 1))
	print('Made V')

	# Get initial loss, so the stopping condition works on first iteration
	# a = loss(train, U, V)
	# print('Calculated loss')
	# errors.append(a)


	# SGD loop
	iterations = 0
	while iterations <= maxIterations:
	# while True:
		iterations += 1

		# Get a random user,rating pair from the data
		i = random.randint(0, M - 1)
		j = random.randint(0, N - 1)

		# Update the U and V at the i-th and j-th columns respectively by subtracting the gradient
		U[:, i] -= eta * (l * U[:, i] - (V[:, j] * (train[i, j] - np.dot(np.transpose(U[:, i]), V[:, j]))))
		V[:, j] -= eta * (l * V[:, j] - (U[:, i] * (train[i, j] - np.dot(np.transpose(U[:, i]), V[:, j]))))

		# Check for error roughly once an epoch
		# if iterations % 100000 == 0:
		# 	errors.append(loss(train, U, V))
		# 	print(errors[-1])
		# 	if iterations == maxIterations or m.fabs(errors[-1] - errors[-2]) / m.fabs(errors[1] - errors[0]) <= epsilon:
		# 		print(iterations)
		# 		break
	print(loss(train, U, V))
	df1 = pd.DataFrame(U)
	print('Made U df')
	df2 = pd.DataFrame(V)
	print('Made V df')
	df1.to_csv('U.csv', index=False, header=False)
	print('Made U.csv')
	df2.to_csv('V.csv', index=False, header=False)
	print('Made V.csv')


if __name__ == '__main__':
	main()
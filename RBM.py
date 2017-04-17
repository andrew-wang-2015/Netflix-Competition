import get_data
from sklearn.neural_network import BernoulliRBM
from sklearn.preprocessing import normalize
from sklearn import linear_model

user_sparse_matrix = get_data.get_train_data_user('../um/small_train.dta')
user_sparse_normed = normalize(user_sparse_matrix, norm='l1', axis=1)

# Train RBM on User-Movie dataset
def RBM_user(sparse_matrix):
	# Get n_components hidden factors for each movie
	model = BernoulliRBM(n_components=200, learning_rate=0.05, n_iter=52, verbose=1)
	model.fit(sparse_matrix)
	print (model.components_.shape)
	return model.components_

def test_RBM():


if __name__ == '__main__':
	RBM_user(user_sparse_normed)
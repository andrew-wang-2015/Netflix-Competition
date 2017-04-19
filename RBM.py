from __future__ import print_function
import get_data_progbar2
import scipy.sparse
from sklearn.neural_network import BernoulliRBM
from sklearn.preprocessing import normalize
from sklearn import linear_model
import numpy as np



# Data is made continuous for continuous RBM
user_sparse_matrix = get_data_progbar2.get_train_data_user('../um/small_train.dta')
#user_sparse_normed = normalize(user_sparse_matrix, norm='l1', axis=1)

class RBM:
  
  def __init__(self, num_visible, num_hidden, learning_rate = 0.1):
    self.num_hidden = num_hidden
    self.num_visible = num_visible
    self.learning_rate = learning_rate

    # Initialize a weight matrix, of dimensions (num_visible x num_hidden), using
    # a Gaussian distribution with mean 0 and standard deviation 0.1.
    self.weights = 0.1 * np.random.randn(self.num_visible, self.num_hidden)    
    # Insert weights for the bias units into the first row and first column.
    self.weights = np.insert(self.weights, 0, 0, axis = 0)
    self.weights = np.insert(self.weights, 0, 0, axis = 1)

  def train(self, data, max_epochs = 1000):
    """
    Train the machine.
    Parameters
    ----------
    data: A matrix where each row is a training example consisting of the states of visible units.    
    """

    num_examples = data.shape[0]

    # Insert bias units of 1 into the first column.
    data = np.insert(data, 0, 1, axis = 1)

    for epoch in range(max_epochs):      
      # Clamp to the data and sample from the hidden units. 
      # (This is the "positive CD phase", aka the reality phase.)
      pos_hidden_activations = np.dot(data, self.weights)  
      pos_hidden_probs = self._logistic(pos_hidden_activations)
      pos_hidden_states = pos_hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
      # Note that we're using the activation *probabilities* of the hidden states, not the hidden states       
      # themselves, when computing associations. We could also use the states; see section 3 of Hinton's 
      # "A Practical Guide to Training Restricted Boltzmann Machines" for more.
      pos_associations = np.dot(data.T, pos_hidden_probs)

      # Reconstruct the visible units and sample again from the hidden units.
      # (This is the "negative CD phase", aka the daydreaming phase.)
      neg_visible_activations = np.dot(pos_hidden_states, self.weights.T)
      neg_visible_probs = self._logistic(neg_visible_activations)
      neg_visible_probs[:,0] = 1 # Fix the bias unit.
      neg_hidden_activations = np.dot(neg_visible_probs, self.weights)
      neg_hidden_probs = self._logistic(neg_hidden_activations)
      # Note, again, that we're using the activation *probabilities* when computing associations, not the states 
      # themselves.
      neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)

      # Update weights.
      self.weights += self.learning_rate * ((pos_associations - neg_associations) / num_examples)

      error = np.sum((data - neg_visible_probs) ** 2)
      print("Epoch %s: error is %s" % (epoch, error))

  def run_visible(self, data):
    """
    Assuming the RBM has been trained (so that weights for the network have been learned),
    run the network on a set of visible units, to get a sample of the hidden units.
    
    Parameters
    ----------
    data: A matrix where each row consists of the states of the visible units.
    
    Returns
    -------
    hidden_states: A matrix where each row consists of the hidden units activated from the visible
    units in the data matrix passed in.
    """
    
    num_examples = data.shape[0]
    
    # Create a matrix, where each row is to be the hidden units (plus a bias unit)
    # sampled from a training example.
    hidden_states = np.ones((num_examples, self.num_hidden + 1))
    
    # Insert bias units of 1 into the first column of data.
    data = np.insert(data, 0, 1, axis = 1)

    # Calculate the activations of the hidden units.
    hidden_activations = np.dot(data, self.weights)
    # Calculate the probabilities of turning the hidden units on.
    hidden_probs = self._logistic(hidden_activations)
    # Turn the hidden units on with their specified probabilities.
    hidden_states[:,:] = hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
    # Always fix the bias unit to 1.
    # hidden_states[:,0] = 1
  
    # Ignore the bias units.
    hidden_states = hidden_states[:,1:]
    return hidden_states
    
  # TODO: Remove the code duplication between this method and `run_visible`?
  def run_hidden(self, data):
    """
    Assuming the RBM has been trained (so that weights for the network have been learned),
    run the network on a set of hidden units, to get a sample of the visible units.
    Parameters
    ----------
    data: A matrix where each row consists of the states of the hidden units.
    Returns
    -------
    visible_states: A matrix where each row consists of the visible units activated from the hidden
    units in the data matrix passed in.
    """

    num_examples = data.shape[0]

    # Create a matrix, where each row is to be the visible units (plus a bias unit)
    # sampled from a training example.
    visible_states = np.ones((num_examples, self.num_visible + 1))

    # Insert bias units of 1 into the first column of data.
    data = np.insert(data, 0, 1, axis = 1)

    # Calculate the activations of the visible units.
    visible_activations = np.dot(data, self.weights.T)
    # Calculate the probabilities of turning the visible units on.
    visible_probs = self._logistic(visible_activations)
    # Turn the visible units on with their specified probabilities.
    visible_states[:,:] = visible_probs > np.random.rand(num_examples, self.num_visible + 1)
    # Always fix the bias unit to 1.
    # visible_states[:,0] = 1

    # Ignore the bias units.
    visible_states = visible_states[:,1:]
    return visible_states
    
  def daydream(self, num_samples):
    """
    Randomly initialize the visible units once, and start running alternating Gibbs sampling steps
    (where each step consists of updating all the hidden units, and then updating all of the visible units),
    taking a sample of the visible units at each step.
    Note that we only initialize the network *once*, so these samples are correlated.
    Returns
    -------
    samples: A matrix, where each row is a sample of the visible units produced while the network was
    daydreaming.
    """

    # Create a matrix, where each row is to be a sample of of the visible units 
    # (with an extra bias unit), initialized to all ones.
    samples = np.ones((num_samples, self.num_visible + 1))

    # Take the first sample from a uniform distribution.
    samples[0,1:] = np.random.rand(self.num_visible)

    # Start the alternating Gibbs sampling.
    # Note that we keep the hidden units binary states, but leave the
    # visible units as real probabilities. See section 3 of Hinton's
    # "A Practical Guide to Training Restricted Boltzmann Machines"
    # for more on why.
    for i in range(1, num_samples):
      visible = samples[i-1,:]

      # Calculate the activations of the hidden units.
      hidden_activations = np.dot(visible, self.weights)      
      # Calculate the probabilities of turning the hidden units on.
      hidden_probs = self._logistic(hidden_activations)
      # Turn the hidden units on with their specified probabilities.
      hidden_states = hidden_probs > np.random.rand(self.num_hidden + 1)
      # Always fix the bias unit to 1.
      hidden_states[0] = 1

      # Recalculate the probabilities that the visible units are on.
      visible_activations = np.dot(hidden_states, self.weights.T)
      visible_probs = self._logistic(visible_activations)
      visible_states = visible_probs > np.random.rand(self.num_visible + 1)
      samples[i,:] = visible_states

    # Ignore the bias units (the first column), since they're always set to 1.
    return samples[:,1:]        
      
  def _logistic(self, x):
    return 1.0 / (1 + np.exp(-x))

# Convert a specific user's ratings to a binary K x M matrix, where K = number of different ratings 
# and M = number of movies that this user rated.
def convert_to_V(user_sm):
  #user_ratings = user_sm.getrow(user_num)
  num_ratings = user_sm.getnnz()
  #movie_ids = np.zeros(num_ratings)
  orig_data = user_sm.data
  orig_row = user_sm.row
  orig_col = user_sm.col
  fin_row = np.zeros(num_ratings)
  #col = np.zeros(num_ratings)
  fin_col = orig_col
  fin_data = np.ones(num_ratings)
  #count = 0
  for i, rating in np.ndenumerate(orig_data):
    fin_row[i] = (orig_row[i]) * 5 + (rating - 1)
    #fin_col[count] = count
    # i is tuple of index in original matrix
    #movie_ids[count] = i[1]
    #count += 1
  return scipy.sparse.coo_matrix((fin_data,(fin_row,fin_col)), shape=(user_sm.shape[0] * 5, user_sm.shape[1]))

def predict_new_rating(hidden_states, weights, visible_bias, user_id, movie_id):
  # Produce vector P_hat[j] of p(h_j = 1 | V) for each factor j
  numerators = np.zeros(5)
  for rating in range(1, 6):
    P_hat = hidden_states[user_id * 5 + (rating - 1)]
    numerators[rating - 1] = visible_bias[user_id * 5 + (rating - 1)]
    for j in range(hidden_states.shape[1]):
      numerators[rating - 1] += P_hat[j] * weights[j][movie_id]
    numerators[rating - 1] = np.exp(numerators[rating - 1])

  denomin = np.sum(numerators)
  final_rate = 0
  for i in range(5):
    print (numerators[i])
    final_rate += numerators[i] * (i + 1) / denomin
  return final_rate


# Train RBM on User-Movie dataset
def RBM_single_user(V):
  # Get n_components hidden factors for each movie
  model = BernoulliRBM(n_components=100, learning_rate=0.0002, n_iter=18, verbose=1)
  model.fit(V)
  print (model.components_.shape)
  return model.components_


if __name__ == '__main__':

  # Data is made continuous for continuous RBM
  user_sparse_matrix = get_data_progbar2.get_train_data_user('../um/small_train.dta')

  user_mat = convert_to_V(user_sparse_matrix)

  print (user_mat.getnnz())

  model = BernoulliRBM(n_components=100, learning_rate=0.02, n_iter=50, verbose=1, batch_size=1000)
  model.fit(user_mat)
  print(model.components_.shape)
  #bias_rates = # model.
  print('Weight Matrix: \n', model.components_)
  print('Visible Biases: \n', model.intercept_visible_)
  print('Hidden Biases: \n', model.intercept_hidden_)
  #print(model.components_)
  hidden_probs = model.transform(user_mat)
  print('Hidden Units: \n', hidden_probs)
  #print()

  print ('New Rating: ', predict_new_rating(hidden_probs, model.components_, model.intercept_visible_, user_sparse_matrix.row[0], user_sparse_matrix.col[0]))
  print ('Actual Rating: ', user_sparse_matrix.data[0])
  #r.train(user_0_mat, max_epochs = 5000)

  #r = RBM(num_visible = 6, num_hidden = 2)
  '''training_data = np.array([[1,1,1,0,0,0],[1,0,1,0,0,0],[1,1,1,0,0,0],[0,0,1,1,1,0], [0,0,1,1,0,0],[0,0,1,1,1,0]])
  r.train(training_data, max_epochs = 5000)
  print(r.weights)
  user = np.array([[0,0,0,1,1,0]])
  print(r.run_visible(user))'''

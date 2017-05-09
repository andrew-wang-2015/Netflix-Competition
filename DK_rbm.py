mimport numpy as np
import os

import zconfig
import utils

class RBM(object):

    """ Restricted Boltzmann Machine implementation using TensorFlow.
    The interface of the class is sklearn-like.
    """

    def __init__(self, num_visible, num_hidden, visible_unit_type='bin', main_dir='rbm',
                 gibbs_sampling_steps=1, learning_rate=0.01, batch_size=10, num_epochs=10, stddev=0.1, verbose=0):

        """
        :param num_visible: number of visible units
        :param num_hidden: number of hidden units
        :param visible_unit_type: type of the visible units (binary or gaussian)
        :param main_dir: main directory to put the models, data and summary directories
        :param model_name: name of the model, used to save data
        :param gibbs_sampling_steps: optional, default 1
        :param learning_rate: optional, default 0.01
        :param batch_size: optional, default 10
        :param num_epochs: optional, default 10
        :param stddev: optional, default 0.1. Ignored if visible_unit_type is not 'gauss'
        :param verbose: level of verbosity. optional, default 0
        """

        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.visible_unit_type = visible_unit_type
        self.main_dir = main_dir
        self.model_name = model_name
        self.gibbs_sampling_steps = gibbs_sampling_steps
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.stddev = stddev
        self.verbose = verbose
        self.W = None
        self.bh_ = None
        self.bv_ = None
        self.b_rating_movie = None
        self.H = None

        '''        self.w_upd8 = None
        self.bh_upd8 = None
        self.bv_upd8 = None

        self.encode = None

        self.loss_function = None

        self.input_data = None
        self.hrand = None
        self.vrand = None
        self.validation_size = None
        '''

    def fit(self, train_set):

        """ Fit the model to the training data.
        :param train_set: training set
        :param validation_set: validation set. optional, default None
        :param restore_previous_model:
                    if true, a previous trained model
                    with the same name of this model is restored from disk to continue training.
        :return: self
        """
        K = 5
        # Initialize a weight matrix, of dimensions (num_visible x num_hidden), using
        # a Gaussian distribution with mean 0 and standard deviation 0.1.
        self.W = 0.1 * np.random.randn(self.num_visible, self.num_hidden, K)    
        # Hidden Unit Bias
        self.bh_ = np.random.randn(self.num_hidden)
        # Bias of rating Movie m Rating r
        self.b_rating_movie = np.random.randn(self.num_visible, K)

        self.h = np.random.randn(self.num_hidden)

        weights_contrib = 

    def calculate_energy(W, bh, b_rating_movie, H):
        weights_contrib = 0
        for i in self.num_visible:
            for j in self.num_hidden:
                for r in K:
                    weights_contrib += W[i][j][r] * H[j] *

    def _logistic(self, x):
    return 1.0 / (1 + np.exp(-x))
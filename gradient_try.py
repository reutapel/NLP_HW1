from datetime import datetime
import math
import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import minimize
import pickle
import time
import os
import copy

# todo: change code


class Gradient(object):
    """
    in this class we implement the tools needed for a known gradient descent of scipy
    according to the data of MEMM with regularization on the weights
    """
    def __init__(self, model, lambda_value):
        self.model = model
        self.v_init = np.zeros(shape=len(model.features_vector), dtype=int)
        self.lambda_value = lambda_value
        self.feature_vector_train = model.history_tag_feature_vector_train
        self.feature_vector_denominator = model.history_tag_feature_vector_denominator
        self.tags_dict = model.tags_dict
        self.index_of_loss = 1
        self.index_gradient = 1
        self.file_name = None
        self.hist_name = None
        self.gradient_per_itter = []
        self.loss_per_itter = []

    def gradient(self, v):
        """
        this methods calculate the gradient of the log-linear problem
        :param v: the weight vector
        :return: the gradient of L(v)
        """
        empirical_counts = csr_matrix(np.zeros(shape=len(v), dtype=int))   # empirical counts
        expected_counts = csr_matrix(np.zeros(shape=len(v), dtype=int))     # expected counts
        weight_vector = np.copy(v)     # weight vector

        for _, feature_vector in self.feature_vector_train.items():
            # multiple in the freq [0] of the history vector (X) with the actual feature vector [1]
            empirical_counts += feature_vector[1] * feature_vector[0]

        for history_tag, feature_vector in self.feature_vector_train.items():
            nominator_dict = {}
            denominator_dict = 0
            feature_freq = 0
            for tag_prime, _ in self.tags_dict.items():
                if (history_tag[0], tag_prime) in self.feature_vector_denominator:
                    # history[0] - x vector
                    feature_freq, feature_vector_current = self.feature_vector_denominator[history_tag[0], tag_prime]
                    inner_exp = math.exp(feature_vector_current.dot(v))
                    denominator_dict += inner_exp
                    nominator_dict[tag_prime] = inner_exp * feature_vector_current

            expected_counts_inner = 0
            for tag_prime, _ in self.tags_dict.items():
                if (history_tag[0], tag_prime) in self.feature_vector_denominator:
                    expected_counts_inner += nominator_dict[tag_prime] / denominator_dict
            expected_counts += expected_counts_inner * feature_freq     # multiple in the freq of the history vector X

        print('{}: finished descent step of gradient #{}'.format(time.asctime(time.localtime(time.time())),
                                                                 self.index_gradient))
        self.index_gradient += 1
        empirical_counts = empirical_counts.toarray()
        expected_counts = expected_counts.toarray()
        gradient = (expected_counts - empirical_counts + self.lambda_value * weight_vector).transpose()
        self.gradient_per_itter.append(np.linalg.norm(gradient))
        return gradient

    def loss(self, v):
        """
        this method calculate the loss on the weight vector
        :param v: weight vector
        :return: the loss of the weight vector, with minus sign
        """

        normalizer_term = 0
        norm_l2 = 0
        linear_term = 0

        norm_l2 += 0.5 * pow(np.linalg.norm(v), 2)  # norm L2 of the feature vector

        for history_tag, feature_vector_list in self.feature_vector_train.items():
            feature_freq, feature_vector = feature_vector_list
            feature_vector_temp = feature_freq * feature_vector     # multiple in freq of history
            linear_term += float(feature_vector_temp.dot(v))  # linear term

            # 2: 1-to-n log of sum of exp. of v*f(x,y') for all y' in Y
            normalize_inner = 0.0
            counter_miss_tag = 0.0
            feature_freq_denominator = 0
            for tag in self.tags_dict:
                if (history_tag[0], tag) in self.feature_vector_denominator:
                    feature_freq_denominator, feature_vector_current = \
                        self.feature_vector_denominator[history_tag[0], tag]
                    dot_product = feature_vector_current.dot(v)
                    normalize_inner += math.exp(dot_product)

                else:
                    counter_miss_tag += 1
            normalizer_term += math.log(normalize_inner) * feature_freq_denominator  # multiple in freq of history

        print('{}: finished loss step #{}'.format(time.asctime(time.localtime(time.time())), self.index_of_loss))
        self.index_of_loss += 1
        loss = normalizer_term + self.lambda_value * norm_l2 - linear_term
        self.loss_per_itter.append(loss)
        return loss

    def gradient_descent(self, file_name=None):
        """
        this method learns the weight vector from the training data
        it performs gradient descent with minus sign which is equivalent to gradient ascent
        :param file_name: wheather we want to retain the weight vector
        :return: weight vector
        """
        if file_name and os.path.isfile(file_name):
            self.file_name = file_name
            return pickle.load(open(file_name, 'rb'))
        result = minimize(method='L-BFGS-B', fun=self.loss, x0=self.v_init, jac=self.gradient,
                          options={'disp': True, 'maxiter': 30, 'ftol': 1e2*np.finfo(float).eps})

        print('finished gradient. res: {0}'.format(result.x))
        file_name = "w_vec_{0.day}_{0.month}_{0.year}_{0.hour}_{0.minute}_{0.second}.pkl".format(datetime.now())
        hist_name = "hist_{0.day}_{0.month}_{0.year}_{0.hour}_{0.minute}_{0.second}.pkl".format(datetime.now())
        self.file_name = os.path.join('resources', file_name)
        pickle.dump(result, open(self.file_name, 'wb'))
        hist_dict = {key: (grad, loss) for key, (grad, loss)
                     in enumerate(zip(self.gradient_per_itter, self.loss_per_itter))}
        self.hist_name = os.path.join('resources', hist_name)
        pickle.dump(hist_dict, open(self.hist_name, 'wb'))
        return result

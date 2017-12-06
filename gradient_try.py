import math
import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import minimize
import pickle
import os

# todo: change code

class Gradient(object):
    """
    in this class we implement the tools needed for a known gradient descent of scipy
    according to the data of MEMM with regularization on the weights
    """
    def __init__(self, memm, lamda):
        self.memm = memm
        self.w_init = np.zeros(shape=len(memm.features_vector), dtype=int)
        self.lamda = lamda
        self.history_tag_feature_vector_train = memm.history_tag_feature_vector_train
        self.history_tag_feature_vector_denominator = memm.history_tag_feature_vector_denominator
        self.tags_dict = memm.tags_dict
        self.iteration_counter = 0
        self.index_of_loss = 1
        self.index_gradient = 1

    def gradient(self, v):
        """
        this methods calculate the gradient of the log-linear problem
        :param v: the weight vector
        :return: the gradient of L(v)
        """
        empirical_counts = csr_matrix(np.zeros(shape=len(v), dtype=int))   # empirical counts
        expected_counts = 0     # expected counts
        weight_vector = np.copy(v)     # weight vector

        for _, feature_vector in self.history_tag_feature_vector_train.items():
            empirical_counts += feature_vector

        for history_tag, feature_vector in self.history_tag_feature_vector_train.items():
            tag_exp_dict = {}
            sum_dict_denominator = 0
            for tag_prime, _ in self.tags_dict.items():
                if (history_tag[0], tag_prime) in self.history_tag_feature_vector_denominator:
                    feature_vector_current = self.history_tag_feature_vector_denominator[history_tag[0], tag_prime]   # history[0] - x vector
                    cur_res = math.exp(feature_vector_current.dot(v))
                    sum_dict_denominator += cur_res
                    tag_exp_dict[tag_prime] = cur_res
                    # todo: whether we can use feaure_vector_current here instead of next loop

            second_part_inner = 0
            for tag_prime, _ in self.tags_dict.items():
                if (history_tag[0], tag_prime) in self.history_tag_feature_vector_denominator:
                    right_var = tag_exp_dict[tag_prime] / sum_dict_denominator
                    second_part_inner += (self.history_tag_feature_vector_denominator[history_tag[0], tag_prime] * right_var)
            expected_counts += second_part_inner

        print('finished descent step of gradient')
        print(self.index_gradient)
        self.index_gradient += 1

        empirical_counts = empirical_counts.toarray()
        expected_counts = expected_counts.toarray()

        return (- empirical_counts + expected_counts + self.lamda * weight_vector).transpose()

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

        for history_tag, feature_vector in self.history_tag_feature_vector_train.items():

            linear_term += float(feature_vector.dot(v))  # linear term

            # 2: 1-to-n log of sum of exp. of v*f(x,y') for all y' in Y
            first_part_inner = 0
            counter_miss_tag = 0
            for tag in self.tags_dict:

                if (history_tag[0], tag) in self.history_tag_feature_vector_denominator:
                    feature_vector_current = self.history_tag_feature_vector_denominator[history_tag[0], tag]
                    cur_res = feature_vector_current.dot(v)
                    first_part_inner += math.exp(cur_res)
                else:
                    counter_miss_tag += 1
            normalizer_term += math.log(first_part_inner)

        print('finished loss step #{0}'.format(self.index_of_loss))
        self.index_of_loss += 1
        return normalizer_term + self.lamda*norm_l2 - linear_term

    def gradient_descent(self, flag=False):
        """
        this method learns the weight vector from the training data
        it performs gradient descent with minus sign which is equivalent to gradient ascent
        :param flag: weather we want to retrain the weight vector
        :return: weight vector
        """
        file_name = os.path.join('resources', 'w_vec.pkl')
        if flag and os.path.isfile(file_name):
            return pickle.load(open(file_name, 'rb'))
        result = minimize(method='L-BFGS-B', fun=self.loss, x0=self.w_init, jac=self.gradient,
                          options={'disp': True, 'maxiter': 15, 'factr': 1e2})
        print('finished gradient. res: {0}'.format(result.x))
        pickle.dump(result, open(file_name, 'wb'))
        return result

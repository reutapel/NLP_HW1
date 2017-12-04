import math
import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import minimize
import pickle


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
        first_part = csr_matrix(np.zeros_like(v))   # empirical counts
        second_part = 0     # expected counts
        third_part = np.copy(v)     # weight vector

        for history_tag, feature_vector in self.history_tag_feature_vector_train.items():
            first_part = np.add(first_part, feature_vector)

        for history_tag, feature_vector in self.history_tag_feature_vector_train.items():

            tag_exp_dict = {}
            sum_dict_denominator = 0
            for tag_prime, flag in self.tags_dict.items():
                if (history_tag[0], tag_prime) in self.history_tag_feature_vector_denominator:
                    feature_vector_current = self.history_tag_feature_vector_denominator[history_tag[0], tag_prime]   # history[0] - x vector
                    cur_res = math.exp(feature_vector_current.dot(v))
                    sum_dict_denominator += cur_res
                    tag_exp_dict[tag_prime] = cur_res

            second_part_inner = 0
            for tag_prime, flag in self.tags_dict.items():
                if (history_tag[0], tag_prime) in self.history_tag_feature_vector_denominator:
                    right_var = tag_exp_dict[tag_prime] / sum_dict_denominator
                    second_part_inner = second_part_inner + (self.history_tag_feature_vector_denominator[history_tag[0], tag_prime] * right_var)
            second_part += second_part_inner

        print('finished descent step of gradient')
        print(self.index_gradient)
        self.index_gradient += 1

        first_part = first_part.toarray()
        second_part = second_part.toarray()

        return (- first_part + second_part + self.lamda * third_part).transpose()

    def loss(self, v):

        first_part = 0
        second_part = 0
        third_part = 0

        for history_tag, feature_vector in self.history_tag_feature_vector_train.items():

            # 3: 1-to-n v*f_v
            third_part += float(feature_vector.dot(v))

            # 1: 1-to-n log of sum of exp. of v*f(x,y') for all y' in Y
            first_part_inner = 0
            counter_miss_tag = 0
            for tag in self.tags_dict:

                if (history_tag[0], tag) in self.history_tag_feature_vector_denominator:
                    feature_vector_current = self.history_tag_feature_vector_denominator[history_tag[0], tag]
                    cur_res = feature_vector_current.dot(v)
                    if cur_res != 0:
                        stop = 5
                    first_part_inner += math.exp(cur_res)
                else:
                    counter_miss_tag +=1
            first_part += math.log(first_part_inner)

        # 2: L2-norm of v
        second_part += 0.5*pow(np.linalg.norm(v), 2)

        print('finished loss step')
        print(self.index_of_loss)
        self.index_of_loss += 1
        return first_part + self.lamda*second_part - third_part

    def gradient_descent(self):

        result = minimize(method='L-BFGS-B', fun=self.loss, x0=self.w_init, jac=self.gradient,
                          options={'disp': True, 'maxiter': 15, 'factr': 1e2})
        print('finished gradient')
        print(result.x)
        pickle.dump(result, open('resources\\w_vec.pkl', 'wb'))
        return result

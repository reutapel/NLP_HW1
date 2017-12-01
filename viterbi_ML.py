import numpy as np
import math
import itertools
import time
from Print_and_save_results import print_save_results
import logging


directory = '\\Users\\reutapel\\Documents\\Technion\\Msc\\NLP\\hw1\\wet/NLP_HW1\\'


class viterbi(object):
    """ Viterbi algorithm for 2-order MEMM model"""
    def __init__(self, model, data_file, use_stop_prob, w):
        self.model = model
        self.transition_mat = {}
        self.emission_mat = {}
        self.states = list(itertools.chain.from_iterable(model.word_tag_dict.values()))
        self.weight = w
        self.predict_file = data_file
        self.word_tag_dict = model.word_tag_dict
        self.use_stop_prob = use_stop_prob  # TODO: check if there is a stop in the sentence and how I need to use it
        self.history_tag_feature_vector = model.history_tag_feature_vector

    def viterbi_all_data(self):
        predict_dict = {}

        with open(self.predict_file, 'r') as predict:
            sentence_index = 0
            for sentence in predict:
                # print('{}: Start viterbi on sentence index {}'.format(time.asctime(time.localtime(time.time())),
                #                                                       sequence_index))
                # parsing of the sequence to word_tag
                # TODO: check how the word_tag in the sentence are splited
                word_tag_list = sentence.split(',')
                # TODO: check which of the following we need
                if '\n' in word_tag_list[len(word_tag_list) - 1]:
                    word_tag_list[len(word_tag_list) - 1] = word_tag_list[len(word_tag_list) - 1].replace('\n', '')
                while ' ' in word_tag_list:
                    word_tag_list.remove(' ')
                while '' in word_tag_list:
                    word_tag_list.remove('')
                while '\n' in word_tag_list:
                    word_tag_list.remove('\n')

                # predict the tags for the specific sentence
                viterbi_results = self.viterbi_sentence(word_tag_list)

                # create a list of word_tag for the prediction of the Viterbi algorithm
                seq_word_tag_predict = []
                for idx_tag, tag in enumerate(viterbi_results):
                    word = word_tag_list[idx_tag].split('_')[0]
                    prediction = str(word + '_' + str(tag))
                    seq_word_tag_predict.append(prediction)

                predict_dict[sentence_index] = seq_word_tag_predict

                sentence_index += 1
            print('{}: prediction for all sentences{}'.format((time.asctime(time.localtime(time.time()))),
                                                              predict_dict))
            logging.info('{}: prediction for all sentences{}'.format((time.asctime(time.localtime(time.time()))),
                                                                     predict_dict))

        return predict_dict

    def viterbi_sentence(self, word_tag_list):
        sen_word_tag_predict = {}

        number_of_words = len(word_tag_list)
        num_states = len(self.states)

        # create pi and bp numpy
        pi = np.ones(shape=(number_of_words+1, num_states, num_states), dtype=float) * float("-inf")
        bp = np.ones(shape=(number_of_words+1, num_states, num_states), dtype='int32') * -1

        # initialization: # will be 0 in the numpy
        pi[0, 0, 0] = 1.0

        # algorithm:
        # k = 1,...,n find the pi and bp for the word in position k
        # TODO: check which parsers I need to do here
        for k in range(1, number_of_words+1):
            if k == 1:  # the word in position 1
                x_k_3, x_k_2, x_k_1 = '#', '#', '#'  # words in k-3, k-2 and in k-1
            elif k == 2:  # the word in position 2
                x_k_1 = word_tag_list[k - 2].split('_')[0]  # word in k-1
                x_k_3, x_k_2 = '#', '#'  # word in k-2
            elif k == 3:  # the word in position 3
                x_k_1 = word_tag_list[k - 2].split('_')[0]  # word in k-1
                x_k_2 = word_tag_list[k - 3].split('_')[0]  # word in k-2
                x_k_3 = '#'
            else:
                x_k_1 = word_tag_list[k - 2].split('_')[0]  # word in k-1
                x_k_2 = word_tag_list[k - 3].split('_')[0]  # word in k-2
                x_k_3 = word_tag_list[k - 4].split('_')[0]  # word in k-3
            if k in range(1, number_of_words-2):
                x_k_p_3 = word_tag_list[k + 2].split('_')[0]  # word k+3
                x_k_p_2 = word_tag_list[k + 1].split('_')[0]  # word k+2
                x_k_p_1 = word_tag_list[k].split('_')[0]      # word k+1
            elif k == number_of_words-2:  # word in position n-2, no word in k+3
                x_k_p_3 = '#'  # word k+3
                x_k_p_2 = word_tag_list[k + 1].split('_')[0]  # word k+2
                x_k_p_1 = word_tag_list[k].split('_')[0]      # word k+1
            elif k == number_of_words-1:  # word in position n-1, no word in k+3 and k+2
                x_k_p_3, x_k_p_2 = '#', '#'  # word k+3 and k+2 and k+1
                x_k_p_1 = word_tag_list[k].split('_')[0]      # word k+1
            else:  # word in position n, no word in k+3 and k+2
                x_k_p_3, x_k_p_2, x_k_p_1 = '#', '#', '#'  # word k+3 and k+2 and k+1
            x_k = word_tag_list[k-1].split('_')[0]
            for u in self.possible_tags(x_k_1):
                for v in self.possible_tags(x_k):
                    calc_max_pi = float("-inf")
                    calc_argmax_pi = -1
                    for w in self.possible_tags(x_k_2):
                        w_u_pi = pi[k - 1, int(w), int(u)]
                        tags_for_matrix = [v, u, w]
                        if '0' in tags_for_matrix:
                            for tag_index, tag in enumerate(tags_for_matrix):
                                if tag == '0':
                                    tags_for_matrix[tag_index] = '#'
                            # if x_k_p_3 == '' or x_k_p_2 == '' or x_k_p_1 == '' \
                            #         or x_k_p_3 == '\n' or x_k_p_2 == '\n' or x_k_p_1 == '\n':
                            #     print('Error: x_p_i is "" or \n')
                        q = self.calc_q(tags_for_matrix[0], tags_for_matrix[1], tags_for_matrix[2], x_k_3, x_k_2,
                                        x_k_1, x_k_p_3, x_k_p_2, x_k_p_1, x_k)
                        calc_pi = w_u_pi * q

                        if calc_pi > calc_max_pi:
                            calc_max_pi = calc_pi
                            calc_argmax_pi = int(w)

                    # print int(u), int(v)
                    # TODO: check if I need to delete the int() - depends on the tags type
                    pi[k, int(u), int(v)] = calc_max_pi  # store the max(pi)
                    bp[k, int(u), int(v)] = calc_argmax_pi  # store the argmax(pi) (bp)

        # print pi[n]
        # print bp[n]

        u = np.unravel_index(pi[number_of_words].argmax(), pi[number_of_words].shape)[0]  # argmax for u in n-1
        v = np.unravel_index(pi[number_of_words].argmax(), pi[number_of_words].shape)[1]  # argmax for v in n

        if v == -1 or u == -1:
            print('Error: v or u value is -1')

        sen_word_tag_predict[number_of_words - 1] = v
        sen_word_tag_predict[number_of_words - 2] = u

        for k in range(number_of_words-2, 0, -1):
            sen_word_tag_predict[k - 1] = bp[k+2, sen_word_tag_predict[k], sen_word_tag_predict[k+1]]

        return sen_word_tag_predict

    def possible_tags(self, word):
        if word == '#':
            return ['0']
        else:
            # get all relevant tags for word
            return self.word_tag_dict.get(word)

    def calc_q(self, v, u, w, x_k_3, x_k_2, x_k_1, x_k_p_3, x_k_p_2, x_k_p_1, x_k):  # calculate q for MEMM model

        sum_denominator = 0
        e_w_dot_history_tag_dict = {}

        # TODO: add the option that the word x_k never seen in the training set
        # and then it will get the most common tags

        for tag in self.word_tag_dict.get(x_k):  # all possible tags for the word x_k
            # history + tag feature vector
            if ((w, u, x_k_3, x_k_2, x_k_1, x_k_p_1, x_k_p_2, x_k_p_3, x_k), tag) in self.history_tag_feature_vector:
                current_history_tag_feature_vector = self.history_tag_feature_vector[(w, u, x_k_3, x_k_2, x_k_1, x_k_p_1,
                                                                                      x_k_p_2, x_k_p_3, x_k), tag]
            else:
                current_history_tag_feature_vector = 1
            #     TODO: create a mechanism to generate a feature vector if it is not exists
            # calculate e^(weight*f(history, tag))
            numerators = math.exp(current_history_tag_feature_vector.dot(self.weight))
            sum_denominator += numerators  # sum for the denominator
            e_w_dot_history_tag_dict[tag] = numerators  # save in order to get e_w_dot_history_tag_dict[v]

        return e_w_dot_history_tag_dict[v] / float(sum_denominator)

import numpy as np
import math
import itertools
import time
import logging


# directory = '/Users/reutapel/Documents/Technion/Msc/NLP/hw1/NLP_HW1/'

class viterbi(object):
    """ Viterbi algorithm for 2-order MEMM model"""
    def __init__(self, model, data_file, w):
        self.model = model
        self.transition_mat = {}
        self.emission_mat = {}
        self.all_tags = model.most_common_tags
        self.tags_indeces_dict = {tag: tag_index + 1 for (tag_index, tag) in enumerate(self.all_tags)}
        self.tags_indeces_dict['#'] = 0
        self.indeces_tags_dict = {tag_index + 1: tag for (tag_index, tag) in enumerate(self.all_tags)}
        self.indeces_tags_dict[0] = '#'
        self.weight = w
        self.predict_file = data_file
        self.word_tag_dict = model.word_tag_dict
        self.history_tag_feature_vector = model.history_tag_feature_vector_denominator
        most_common_tags_to_use = model.most_common_tags
        if 'DT' in most_common_tags_to_use:
            most_common_tags_to_use.remove('DT')
        if 'IN' in most_common_tags_to_use:
            most_common_tags_to_use.remove('IN')
        self.most_common_tags = most_common_tags_to_use[:5]
        # all the words that has not seen in the train, but seen in the test in the format: [sen_index, word_index]
        self.unseen_words = []

    @property
    def viterbi_all_data(self):
        predict_dict = {}

        with open(self.predict_file, 'r') as predict:
            sentence_index = 0
            for sentence in predict:
                # print('{}: Start viterbi on sentence index {}'.format(time.asctime(time.localtime(time.time())),
                #                                                       sequence_index))
                # parsing of the sequence to word_tag
                sentence = sentence.rstrip('\n')
                word_tag_list = sentence.split(' ')

                # predict the tags for the specific sentence
                viterbi_results = self.viterbi_sentence(word_tag_list, sentence_index)

                # create a list of word_tag for the prediction of the Viterbi algorithm
                seq_word_tag_predict = [None] * len(word_tag_list)
                for idx_tag, tag in viterbi_results.items():
                    word = word_tag_list[idx_tag].split('_')[0]
                    prediction = str(word + '_' + str(tag))
                    seq_word_tag_predict[idx_tag] = prediction

                predict_dict[sentence_index] = seq_word_tag_predict

                sentence_index += 1
            print('{}: prediction for all sentences{}'.format((time.asctime(time.localtime(time.time()))),
                                                              predict_dict))
            logging.info('{}: prediction for all sentences{}'.format((time.asctime(time.localtime(time.time()))),
                                                                     predict_dict))

        return predict_dict, self.unseen_words

    def viterbi_sentence(self, word_tag_list, sentence_index):
        sen_word_tag_predict = {}

        number_of_words = len(word_tag_list)
        number_of_tags = len(self.all_tags) + 1  # +1 for the tag '#'

        # create pi and bp numpy
        pi = np.ones(shape=(number_of_words+1, number_of_tags, number_of_tags), dtype=float) * float("-inf")
        bp = np.ones(shape=(number_of_words+1, number_of_tags, number_of_tags), dtype='int32') * -1

        # initialization: # will be 0 in the numpy
        pi[0, self.tags_indeces_dict['#'], self.tags_indeces_dict['#']] = 1.0

        # algorithm:
        # k = 1,...,n find the pi and bp for the word in position k
        for k in range(1, number_of_words+1):
            if k == 1:  # the word in position 1
                first_word, second_word = '#', '#'  # words in k-2 and in k-1
            elif k == 2:  # the word in position 2
                second_word = word_tag_list[k - 2].split('_')[0]  # word in k-1
                first_word = '#'  # word in k-2
            elif k == 3:  # the word in position 3
                second_word = word_tag_list[k - 2].split('_')[0]  # word in k-1
                first_word = word_tag_list[k - 3].split('_')[0]  # word in k-2
            else:
                second_word = word_tag_list[k - 2].split('_')[0]  # word in k-1
                first_word = word_tag_list[k - 3].split('_')[0]  # word in k-2
            if k in range(1, number_of_words):
                plus_one_word = word_tag_list[k].split('_')[0]      # word k+1
            else:  # word in position n, no word in k+1
                plus_one_word = '#'  # word in position k+1
            current_word = word_tag_list[k - 1].split('_')[0]
            current_word_possible_tags, unseen_word = self.possible_tags(current_word)
            if unseen_word:  # never the seen the word in the train set
                self.unseen_words.append([sentence_index, k - 1])  # insert the sen_index and the word_index
            for u in self.possible_tags(second_word)[0]:
                for v in current_word_possible_tags:
                    calc_max_pi = float("-inf")
                    calc_argmax_pi = -1
                    for w in self.possible_tags(first_word)[0]:
                        w_u_pi = pi[k - 1, self.tags_indeces_dict[w], self.tags_indeces_dict[u]]
                        q = self.calc_q(v, w, u, second_word, plus_one_word, current_word)
                        calc_pi = w_u_pi * q

                        if calc_pi > calc_max_pi:
                            calc_max_pi = calc_pi
                            calc_argmax_pi = w

                    # print int(u), int(v)
                    calc_argmax_pi = self.tags_indeces_dict[calc_argmax_pi]
                    pi[k, self.tags_indeces_dict[u], self.tags_indeces_dict[v]] = calc_max_pi  # store the max(pi)
                    bp[k, self.tags_indeces_dict[u], self.tags_indeces_dict[v]] = calc_argmax_pi  # store the argmax(pi) (bp)

        # print pi[n]
        # print bp[n]

        u_index = np.unravel_index(pi[number_of_words].argmax(), pi[number_of_words].shape)[0]  # argmax for u in n-1
        v_index = np.unravel_index(pi[number_of_words].argmax(), pi[number_of_words].shape)[1]  # argmax for v in n

        sen_word_tag_predict[number_of_words - 1] = self.indeces_tags_dict[v_index]
        sen_word_tag_predict[number_of_words - 2] = self.indeces_tags_dict[u_index]

        for k in range(number_of_words-2, 0, -1):
            sen_word_tag_predict[k - 1] = self.indeces_tags_dict[bp[k+2,
                                                                    self.tags_indeces_dict[sen_word_tag_predict[k]],
                                                                    self.tags_indeces_dict[sen_word_tag_predict[k+1]]]]

        return sen_word_tag_predict

    def possible_tags(self, word):
        unseen_word = False
        if word == '#':
            return [['#'], unseen_word]
        else:
            # if we never see the current word in the train
            if word not in self.word_tag_dict:
                tags_list = self.most_common_tags
                unseen_word = True
            else:
                tags_list = list(self.word_tag_dict.get(word).keys())
                tags_list = tags_list[1:]

            return [tags_list, unseen_word]

    def calc_q(self, v, second_tag, first_tag, second_word, plus_one_word, current_word):  # calculate q for MEMM model

        sum_denominator = 0
        e_w_dot_history_tag_dict = {}

        # get a list of all possible tags for the current word
        tags_list = self.possible_tags(current_word)[0]

        for tag in tags_list:  # all possible tags for the word x_k
            # history + possible tag we want to check
            if ((first_tag, second_tag, second_word, plus_one_word, current_word), tag) in self.history_tag_feature_vector:
                current_history_tag_feature_vector = self.history_tag_feature_vector[(first_tag, second_tag,
                                                                                      second_word, plus_one_word,
                                                                                      current_word), tag][1]
            # if the history and the tag do not exists in the history_tag_feature_vector from the MEMM
            else:
                current_history_tag_feature_vector = self.model.calculate_history_tag_indexes(first_tag, second_tag,
                                                                                              second_word, plus_one_word,
                                                                                              current_word, tag)
            # calculate e^(weight*f(history, tag))
            numerators = math.exp(current_history_tag_feature_vector.dot(self.weight))
            sum_denominator += numerators  # sum for the denominator
            e_w_dot_history_tag_dict[tag] = numerators  # save in order to get e_w_dot_history_tag_dict[v]

        return e_w_dot_history_tag_dict[v] / float(sum_denominator)

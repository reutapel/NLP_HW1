import numpy as np
import math
import os
import time
import logging
import copy
import csv
import string


# directory = '/Users/reutapel/Documents/Technion/Msc/NLP/hw1/NLP_HW1/'

class viterbi(object):
    """ Viterbi algorithm for 2-order MEMM model"""
    def __init__(self, model, data_file, w):
        self.model = model
        self.transition_mat = {}
        self.emission_mat = {}
        self.all_tags = copy.copy(model.most_common_tags)
        self.number_of_tags = len(self.all_tags) + 2  # +1 for the tag '*' and +1 for the tag 'UNK'
        self.tags_indexes_dict = {tag: tag_index + 1 for (tag_index, tag) in enumerate(self.all_tags)}
        self.tags_indexes_dict['*'] = 0
        self.tags_indexes_dict['UNK'] = self.number_of_tags - 1
        self.indexes_tags_dict = {tag_index + 1: tag for (tag_index, tag) in enumerate(self.all_tags)}
        self.indexes_tags_dict[0] = '*'
        self.indexes_tags_dict[self.number_of_tags - 1] = 'UNK'
        self.weight = w
        self.predict_file = data_file
        self.word_tag_dict = model.word_tag_dict
        self.history_tag_feature_vector = model.history_tag_feature_vector_denominator
        most_common_tags_to_use = copy.copy(model.most_common_tags)
        if 'DT' in most_common_tags_to_use:
            most_common_tags_to_use.remove('DT')
        if 'IN' in most_common_tags_to_use:
            most_common_tags_to_use.remove('IN')
        if ',' in most_common_tags_to_use:
            most_common_tags_to_use.remove(',')
        if '.' in most_common_tags_to_use:
            most_common_tags_to_use.remove('.')
        if '*' in most_common_tags_to_use:
            most_common_tags_to_use.remove('*')
        self.most_common_tags = most_common_tags_to_use[:5]
        self.most_common_tags = ['CD', 'JJ', 'NN', 'NNP', 'NNS']
        # print('most common tags are: {}'.format(self.most_common_tags))
        # all the words that has not seen in the train, but seen in the test in the format: [sen_index, word_index]
        self.unseen_words_indexes = list()
        self.unseen_words = dict()
        self.transition_tag_dict = model.transition_tag_dict
        self.verb_tags_list = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
        self.lower_case = 0
        self.plural_case = 0
        self.d_case = 0
        self.ing_case = 0
        self.n_case = 0
        self.upper_case = 0
        self.digit_case = 0
        self.single_case = 0
        self.unk_case = 0
        self.invalidChars = set(string.punctuation)

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
                    prediction = word + '_' + tag
                    seq_word_tag_predict[idx_tag] = prediction
                    if [sentence_index, idx_tag] in self.unseen_words_indexes:
                        self.unseen_words[(sentence_index, idx_tag)] = self.unseen_words[(sentence_index, idx_tag)] +\
                                                                       '_' + tag

                predict_dict[sentence_index] = seq_word_tag_predict

                sentence_index += 1

        print('Unseen cases number: \nlower_case = {}\nplural_case = {}\nd_case = {}\ning_case = {}\nn_case = {}\n'
              'upper_case = {}\ndigit_case = {}\nsingle_case = {}\nunk_case = {}'.
              format(self.lower_case, self.plural_case, self.d_case, self.ing_case, self.n_case, self.upper_case,
                     self.digit_case, self.single_case, self.unk_case))
        logging.info('Unseen cases number: \nlower_case = {}\nplural_case = {}\nd_case = {}\ning_case = {}\n'
                     'n_case = {}\n''upper_case = {}\ndigit_case = {}\nsingle_case = {}\nunk_case = {}'.
                     format(self.lower_case, self.plural_case, self.d_case, self.ing_case, self.n_case, self.upper_case,
                            self.digit_case, self.single_case, self.unk_case))

        # save the unseen words to file
        unseen_file_name = os.path.join('analysis', 'unseen_words.csv')
        unseen_file = csv.writer(open(unseen_file_name, "w"))
        for key, val in self.unseen_words.items():
            unseen_file.writerow([key, val])

        return predict_dict, self.unseen_words_indexes

    def viterbi_sentence(self, word_tag_list, sentence_index):
        sen_word_tag_predict = {}

        number_of_words = len(word_tag_list)

        # create pi and bp numpy
        pi = np.ones(shape=(number_of_words+1, self.number_of_tags, self.number_of_tags), dtype=float) * float("-inf")
        bp = np.ones(shape=(number_of_words+1, self.number_of_tags, self.number_of_tags), dtype='int32') * -1

        # initialization: # will be 0 in the numpy
        pi[0, self.tags_indexes_dict['*'], self.tags_indexes_dict['*']] = 1.0

        # algorithm:
        # k = 1,...,n find the pi and bp for the word in position k
        for k in range(1, number_of_words+1):
            if k == 1:  # the word in position 1
                first_word, second_word = '*', '*'  # words in k-2 and in k-1
            elif k == 2:  # the word in position 2
                second_word = word_tag_list[k - 2].split('_')[0]  # word in k-1
                first_word = '*'  # word in k-2
            elif k == 3:  # the word in position 3
                second_word = word_tag_list[k - 2].split('_')[0]  # word in k-1
                first_word = word_tag_list[k - 3].split('_')[0]  # word in k-2
            else:
                second_word = word_tag_list[k - 2].split('_')[0]  # word in k-1
                first_word = word_tag_list[k - 3].split('_')[0]  # word in k-2
            if k in range(1, number_of_words):
                plus_one_word = word_tag_list[k].split('_')[0]      # word k+1
            else:  # word in position n, no word in k+1
                plus_one_word = '*'  # word in position k+1
            current_word = word_tag_list[k - 1].split('_')[0]
            current_word_possible_tags, unseen_word = self.possible_tags(current_word, is_cur_word=True)
            if unseen_word:  # never the seen the word in the train set
                self.unseen_words_indexes.append([sentence_index, k - 1])  # insert the sen_index and the word_index
                self.unseen_words[(sentence_index, k - 1)] = word_tag_list[k - 1]
            for u in self.possible_tags(second_word)[0]:
                for v in current_word_possible_tags:
                    calc_max_pi = float("-inf")
                    calc_argmax_pi = -1
                    for w in self.possible_tags(first_word)[0]:
                        w_u_pi = pi[k - 1, self.tags_indexes_dict[w], self.tags_indexes_dict[u]]
                        q = self.calc_q(v, w, u, second_word, plus_one_word, current_word, current_word_possible_tags)
                        calc_pi = w_u_pi * q

                        if calc_pi > calc_max_pi:
                            calc_max_pi = calc_pi
                            calc_argmax_pi = w

                    # print int(u), int(v)
                    calc_argmax_pi = self.tags_indexes_dict[calc_argmax_pi]
                    pi[k, self.tags_indexes_dict[u], self.tags_indexes_dict[v]] = calc_max_pi  # store the max(pi)
                    bp[k, self.tags_indexes_dict[u], self.tags_indexes_dict[v]] = calc_argmax_pi  # store the argmax(pi) (bp)

        # print pi[n]
        # print bp[n]

        u_index = np.unravel_index(pi[number_of_words].argmax(), pi[number_of_words].shape)[0]  # argmax for u in n-1
        v_index = np.unravel_index(pi[number_of_words].argmax(), pi[number_of_words].shape)[1]  # argmax for v in n

        sen_word_tag_predict[number_of_words - 1] = self.indexes_tags_dict[v_index]
        sen_word_tag_predict[number_of_words - 2] = self.indexes_tags_dict[u_index]

        for k in range(number_of_words-2, 0, -1):
            sen_word_tag_predict[k - 1] = self.indexes_tags_dict[bp[k+2,
                                                                    self.tags_indexes_dict[sen_word_tag_predict[k]],
                                                                    self.tags_indexes_dict[sen_word_tag_predict[k+1]]]]
        for k in range(number_of_words):
            if sen_word_tag_predict[k] == 'UNK':
                if k == 0:
                    tag_k_2, tag_k_1 = '*', '*'
                elif k == 1:
                    tag_k_2, tag_k_1 = '*', sen_word_tag_predict[k-1]
                else:
                    tag_k_2, tag_k_1 = sen_word_tag_predict[k - 2], sen_word_tag_predict[k - 1]
                u_v = tag_k_2 + '_' + tag_k_1
                if u_v in self.transition_tag_dict:
                    sen_word_tag_predict[k] = self.transition_tag_dict[u_v]['TOP'][0]
                elif tag_k_1 in self.transition_tag_dict:
                    sen_word_tag_predict[k] = self.transition_tag_dict[tag_k_1]['TOP'][0]
                else:
                    print('tag {} is not in transition_tag_dict'.format(tag_k_1))
                    sen_word_tag_predict[k] = self.most_common_tags[0]  # give th most common tag

        return sen_word_tag_predict

    def possible_tags(self, word, is_cur_word=False):
        unseen_word = False
        if word == '*':
            return [['*'], unseen_word]
        else:
            # if we never see the current word in the train
            if word not in self.word_tag_dict:
                if word == 'Marilyn':
                    reut = 1
                unseen_word = True
                if word.lower() in self.word_tag_dict:  # if the word in unseen, but the lower case of the word is seen
                    tags_list = list(self.word_tag_dict.get(word.lower()).keys())
                    # drop the COUNT cell
                    tags_list = tags_list[1:]
                    if tags_list != ['UNK'] and is_cur_word:
                        # print('use lower case for word {}'.format(word))
                        self.lower_case += 1
                    # print('the word {} is unseen but the word {} is seen'. format(word, word.lower()))
                # word contains number instances or word is a number
                elif any(char.isdigit() for char in word) and not word.isdigit() or word.isdigit():
                    tags_list = ['CD']
                    # print('use word contains number instances or word is a number case for word {}'.format(word))
                    if is_cur_word:
                        self.digit_case += 1
                # word contains upper instances or only upper
                elif (not word.islower() and not word.isupper() or word.isupper())\
                        and (word not in self.invalidChars and not word.isdigit()):
                    tags_list = ['NNP']
                    if is_cur_word:
                        self.upper_case += 1
                        # print('use word contains upper instances or only upper case for word {}'.format(word))
                # word ends with s
                elif word[-1:] == 's':
                    tags_list = self.find_tags_for_unseen_plural_nouns(word)
                    if tags_list != ['UNK'] and is_cur_word:
                        # print('use s case for word {}'.format(word))
                        self.plural_case += 1
                # word ends with d
                elif word[-1:] == 'd':
                    tags_list = self.find_tags_for_unseen_past_verbs(word)
                    if tags_list != ['UNK'] and is_cur_word:
                        self.d_case += 1
                        # print('use d case for word {}'.format(word))
                # word ends with ing
                elif word[-3:] == 'ing':
                    tags_list = self.find_tags_for_unseen_VBG(word)
                    if tags_list != ['UNK'] and is_cur_word:
                        self.ing_case += 1
                        # print('use ing case for word {}'.format(word))
                # word ends with ing
                elif word[-1:] == 'n':
                    tags_list = self.find_tags_for_unseen_VBN(word)
                    if tags_list != ['UNK'] and is_cur_word:
                        self.n_case += 1
                        # print('use n case for word {}'.format(word))
                else:
                    tags_list = self.find_tags_for_unseen_singular_nouns(word)
                    if tags_list != ['UNK'] and is_cur_word:
                        self.single_case += 1
                        # print('use single nouns case for word {}'.format(word))
                if tags_list == ['UNK']:
                    tags_list = self.find_tags_for_unseen_singular_nouns(word)
                    if tags_list != ['UNK'] and is_cur_word:
                        self.single_case += 1
                        # print('use single nouns case for word {}'.format(word))

            else:  # word was seen in train - take the tags seen for it in the train
                tags_list = list(self.word_tag_dict.get(word).keys())
                # drop the COUNT cell
                tags_list = tags_list[1:]

            if tags_list == ['UNK'] and is_cur_word:
                print('UNK tag for word: {}'.format(word))
                self.unk_case += 1

            return [tags_list, unseen_word]

    # check if the singular form of the word is seen. return UNK if not
    def find_tags_for_unseen_plural_nouns(self, word):
        tags_list = ['UNK']
        no_s_word = word[:len(word)-1].lower()
        if no_s_word in self.word_tag_dict:  # check words like boat and boats
            tags_list = list(self.word_tag_dict.get(no_s_word).keys())
            tags_list = tags_list[1:]
        elif word[-2:] == 'es':  # check words like bus and buses
            no_es_word = word[:len(word) - 2]
            if no_es_word in self.word_tag_dict:
                tags_list = list(self.word_tag_dict.get(no_es_word).keys())
                tags_list = tags_list[1:]
        elif word[-2:] == 'ies':  # check words like penny and pennies
            no_ies_word = word[:len(word) - 3]
            if no_ies_word in self.word_tag_dict:
                tags_list = list(self.word_tag_dict.get(no_ies_word).keys())
                tags_list = tags_list[1:]

        if 'NN' in tags_list:  # add VBZ for verbs like wish-wishes
            tags_list = ['NNS', 'VBZ']
        elif 'NNP' in tags_list:
            tags_list = ['NNPS', 'VBZ']
        elif any(i in tags_list for i in self.verb_tags_list):  # for verbs like wish-wishes
            tags_list = ['VBZ']
        else:
            tags_list = ['UNK']

        return tags_list

    # check if the plural form of the word is seen. return UNK if not
    def find_tags_for_unseen_singular_nouns(self, word):
        tags_list = ['UNK']
        s_word = word.lower() + 's'
        es_word = word.lower() + 'es'
        ies_word = word.lower() + 'ies'
        if s_word in self.word_tag_dict:  # check words like boat and boats
            tags_list = list(self.word_tag_dict.get(s_word).keys())
        elif es_word in self.word_tag_dict:  # check words like boat and boats
            tags_list = list(self.word_tag_dict.get(es_word).keys())
        elif ies_word in self.word_tag_dict:  # check words like boat and boats
            tags_list = list(self.word_tag_dict.get(ies_word).keys())

        if 'NNS' in tags_list:  # add VB for verbs like wish-wishes
            tags_list = ['NN', 'VB']
        elif 'NNPS' in tags_list:
            tags_list = ['NNP', 'VB']
        elif any(i in tags_list for i in self.verb_tags_list):
            tags_list = ['VB', 'NNP']
        else:
            tags_list = ['UNK']

        return tags_list

    # check if the present form of the past verb is seen. return UNK if not
    def find_tags_for_unseen_past_verbs(self, word):
        tags_list = ['UNK']
        no_d_word = word[:len(word)-1].lower()
        if no_d_word in self.word_tag_dict:  # check words like bake and baked
            tags_list = list(self.word_tag_dict.get(no_d_word).keys())
        elif word[-2:] == 'ed':
            no_ed_word = word[:len(word)-2]
            if no_ed_word in self.word_tag_dict:  # check words like work and worked
                tags_list = list(self.word_tag_dict.get(no_ed_word).keys())
        elif word[-3:] == 'ied':
            no_ied_y_word = word[:len(word) - 3] + 'y'
            if no_ied_y_word in self.word_tag_dict:  # check words like apply and applied
                tags_list = list(self.word_tag_dict.get(no_ied_y_word).keys())

        if any(i in tags_list for i in self.verb_tags_list):
            tags_list = ['VBN', 'VBD', 'VB', 'JJ']  # add JJ for words like unresolved
        else:
            tags_list = ['UNK']

        return tags_list

    # check if the no ing form of the verb is seen. return UNK if not
    def find_tags_for_unseen_VBG(self, word):
        tags_list = ['UNK']
        no_ing_word = word[:len(word)-3].lower()
        no_ing_e_word = word[:len(word)-3].lower() + 'e'  # for verbs like write and writing
        no_ing_ie_word = word[:len(word)-4].lower() + 'ie'  # for verbs like die and dying
        no_double_last_letter = word[:len(word)-4].lower()  # for verbs like bag and bagging
        if word[-4:] == 'ying':
            if no_ing_ie_word in self.word_tag_dict:  # for verbs like die and dying
                tags_list = list(self.word_tag_dict.get(no_ing_ie_word).keys())
        elif len(word) > 4 and word[-4] == word[-5]:
            if no_double_last_letter in self.word_tag_dict:  # for verbs like bag and bagging
                tags_list = list(self.word_tag_dict.get(no_double_last_letter).keys())
        elif no_ing_word in self.word_tag_dict:  # check words like read and reading
            tags_list = list(self.word_tag_dict.get(no_ing_word).keys())
        elif no_ing_e_word in self.word_tag_dict:  # for verbs like write and writing
            tags_list = list(self.word_tag_dict.get(no_ing_e_word).keys())

        if any(i in tags_list for i in self.verb_tags_list):
            tags_list = ['VBG']
        else:
            tags_list = ['UNK']

        return tags_list

    def find_tags_for_unseen_VBN(self, word):
        tags_list = ['UNK']
        no_n_word = word[:len(word)-1].lower()
        no_en_word = word[:len(word) - 2].lower()
        no_double_last_letter = word[:len(word) - 3].lower()
        if no_n_word in self.word_tag_dict:  # check words like know and known
            tags_list = list(self.word_tag_dict.get(no_n_word).keys())
        elif no_en_word in self.word_tag_dict:  # for verbs like be and been
            tags_list = list(self.word_tag_dict.get(no_en_word).keys())
        elif len(word) > 3 and word[-3] == word[-4]:
            if no_double_last_letter in self.word_tag_dict:  # for verbs like forgot and forgotten
                tags_list = list(self.word_tag_dict.get(no_double_last_letter).keys())

        if any(i in tags_list for i in self.verb_tags_list):
            tags_list = ['VBN']
        else:
            tags_list = ['UNK']

        return tags_list

    # calculate q for MEMM model
    def calc_q(self, v, second_tag, first_tag, second_word, plus_one_word, current_word, current_word_possible_tags):

        sum_denominator = 0
        e_w_dot_history_tag_dict = {}

        for tag in current_word_possible_tags:  # all possible tags for the word x_k
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

import numpy as np
import time
import math
import itertools
from collections import Counter
import csv
from datetime import datetime
from Print_and_save_results import print_save_results
import logging


directory = 'C:\\gitprojects\\ML_PROJECT\\'

class viterbi(object):
    """ Viterbi algorithm for 2-order HMM and MEMM model"""
    def __init__(self, model, model_type, data_file, is_log, use_stop_prob, phase_number=1, use_majority_vote=False,
                 w=0, prediction_for_phase2=None, use_majority2=False):
        # model will be HMM or MEMM object, model_type in ['hmm','memm']
        self.model_type = model_type
        self.model = model
        self.phase_number = phase_number
        self.prediction_for_phase2 = prediction_for_phase2  # dictionary with prediction for the first base in each seq
        if model_type == 'hmm':
            self.transition_mat = model.transition_mat
            self.emission_mat = model.emission_mat
        else:
            self.transition_mat = {}
            self.emission_mat = {}
        self.states = list(itertools.chain.from_iterable(model.word_tag_dict.values()))
        self.weight = w
        self.predict_file = data_file
        self.word_tag_dict = model.word_tag_dict
        self.use_stop_prob = use_stop_prob
        self.use_majority_vote = use_majority_vote
        self.use_majority2 = use_majority2
        if self.use_majority_vote:
            self.number_of_dicts = 3  # number of predictions of first base and "move" seq left
        else:
            self.number_of_dicts = 1
        if model_type == 'memm':
            self.history_tag_feature_vector = model.history_tag_feature_vector
        else:
            self.history_tag_feature_vector = {}
        if is_log:
            self.transition_mat = {key: math.log10(value) for key, value in self.transition_mat.items()}

    def viterbi_all_data(self, chrome='1'):
        predict_dict = {}

        with open(self.predict_file, 'r') as predict:
            sequence_index = 0
            for sequence in predict:
                # print('{}: Start viterbi on sequence index {}'.format(time.asctime(time.localtime(time.time())),
                #                                                       sequence_index))
                word_tag_list = sequence.split(',')
                if '\n' in word_tag_list[len(word_tag_list) - 1]:
                    word_tag_list[len(word_tag_list) - 1] = word_tag_list[len(word_tag_list) - 1].replace('\n', '')
                while ' ' in word_tag_list:
                    word_tag_list.remove(' ')
                while '' in word_tag_list:
                    word_tag_list.remove('')
                while '\n' in word_tag_list:
                    word_tag_list.remove('\n')

                if self.phase_number == 1:
                    n = len(word_tag_list)
                    majority_vote_dict = {}
                    word_tag_list_to_predict = word_tag_list

                    for i in range(0, self.number_of_dicts):
                        majority_vote_dict[(sequence_index, i)] =\
                            self.viterbi_sequence(word_tag_list_to_predict, sequence_index)
                        word_tag_list_to_predict = word_tag_list_to_predict[1:len(word_tag_list_to_predict)]
                    if self.use_majority_vote:
                        most_common_tags = range(n)
                        for word_index in range(2, n):
                            compare_list = []
                            for predict_dictionary in range(0, self.number_of_dicts):
                                compare_list.append(majority_vote_dict[(sequence_index, predict_dictionary)]
                                                    [word_index - predict_dictionary])
                                # list_index - predict_dict: to get to the right place in the relevant dictionary
                                # (because the word get different index in each dictionary)
                            count = Counter(compare_list)
                            most_common_tags[word_index] = count.most_common()[0][0]
                        # predict the first two tags:
                        word_tag_1 = word_tag_list[1].split('_')
                        word_tag_0 = word_tag_list[0].split('_')
                        if most_common_tags[2] in range(1, 5):
                            most_common_tags[1] = int(self.word_tag_dict[word_tag_1[0]][0])
                            most_common_tags[0] = int(self.word_tag_dict[word_tag_0[0]][0])
                        elif most_common_tags[2] in range(5, 9):
                            most_common_tags[1] = int(self.word_tag_dict[word_tag_1[0]][1])
                            most_common_tags[0] = int(self.word_tag_dict[word_tag_0[0]][1])
                        else:
                            print('Error: prediction is not in (1,8)')
                    else:
                        most_common_tags = majority_vote_dict[(sequence_index, 0)].values()

                    seq_word_tag_predict = []
                    seq_word_tag_predict_majority = {}
                    for idx_tag, tag in enumerate(most_common_tags):
                        if tag == 0 or tag == '0' or tag == -1 or tag == '-1':
                            print('Error: tag is: {}'.format(tag))
                        word = word_tag_list[idx_tag].split('_')[0]
                        prediction = str(word + '_' + str(tag))
                        seq_word_tag_predict.append(prediction)
                        if self.use_majority_vote:  # using majority vote
                            for predict_dictionary in range(0, self.number_of_dicts):
                                if predict_dictionary == (self.number_of_dicts-2) and idx_tag < predict_dictionary \
                                        or predict_dictionary == (self.number_of_dicts-1) and idx_tag < predict_dictionary:
                                    if idx_tag == 0:  # create the list
                                        seq_word_tag_predict_majority[(sequence_index, predict_dictionary)] = \
                                            ['not_relevant']
                                        continue
                                    elif idx_tag < predict_dictionary:
                                        # if idx_tag < predict_dictionary there is no prediction
                                        seq_word_tag_predict_majority[(sequence_index, predict_dictionary)].\
                                            append('not_relevant')
                                        continue
                            predict = majority_vote_dict[(sequence_index, predict_dictionary)][idx_tag - predict_dictionary]
                            prediction = str(word + '_' + str(predict))
                            if idx_tag == 0:
                                seq_word_tag_predict_majority[(sequence_index, predict_dictionary)] = [prediction]
                            else:
                                seq_word_tag_predict_majority[(sequence_index, predict_dictionary)].append(prediction)

                    predict_dict[sequence_index] = seq_word_tag_predict

                    seq_word_tag_predict_majority[(sequence_index, self.number_of_dicts)] = seq_word_tag_predict
                    seq_word_tag_predict_majority[(sequence_index, self.number_of_dicts + 1)] = word_tag_list

                    self.write_majority_doc(chrome, seq_word_tag_predict_majority, sequence_index)
                    print('{}: prediction for sequence index {}'.format((time.asctime(time.localtime(time.time()))),
                                                                        sequence_index))
                elif self.phase_number == 2:
                    viterbi_results = self.viterbi_sequence(word_tag_list, sequence_index)
                    seq_word_tag_predict = []
                    seq_word_tag_predict_majority = {}
                    for idx_tag, tag in viterbi_results.items():
                        if tag == 0 or tag == '0' or tag == -1 or tag == '-1':
                            print('Error: tag is: {}'.format(tag))
                        word = word_tag_list[idx_tag].split('_')[0]
                        prediction = str(word + '_' + str(tag))
                        seq_word_tag_predict.append(prediction)
                    predict_dict[sequence_index] = seq_word_tag_predict
                    # save predict and real sequence to csv
                    seq_word_tag_predict_majority[(sequence_index, 0)] = seq_word_tag_predict
                    seq_word_tag_predict_majority[(sequence_index, 1)] = word_tag_list
                    self.write_majority_doc(chrome, seq_word_tag_predict_majority, sequence_index)

                sequence_index += 1
            # print '{}: prediction for all sequences{}'.format((time.asctime(time.localtime(time.time()))),
            #                                                   predict_dict)
        # second solution with majority: predict on the first and than continue --> return only first base predict
        if self.phase_number == 1 and self.use_majority2:
            # return dictionary: {sequence_index:prediction for the first base}
            if self.model_type == 'memm':
                write_file_name = datetime.now().strftime \
                    (directory + 'file_results\\chr' + chrome + '_resultMEMM_%d_%m_%Y_%H_%M.csv')
                confusion_file_name = datetime.now().strftime \
                    (directory + 'confusion_files\\chr' + chrome + '_CMMEMM_%d_%m_%Y_%H_%M.xls')
                seq_confusion_file_name = datetime.now().strftime \
                    (directory + 'confusion_files\\chr' + chrome + '_sqeCMMEMM_%d_%m_%Y_%H_%M.xls')
                # seq_labels_file_name = 'C:/gitprojects/ML project/samples_small_data/chr1_sample_label.xlsx'
                seq_labels_file_name = directory + 'sample_labels150\\chr' + chrome + '_sample_label.xlsx'
                evaluate_class = print_save_results(self.model, 'memm', self.predict_file, predict_dict, write_file_name,
                                                    confusion_file_name, seq_labels_file_name, seq_confusion_file_name)
                word_results_dictionary, seq_results_dictionary = evaluate_class.run()

                logging.info('{}: Related results files are: \n {} \n {} \n {}'.
                             format(time.asctime(time.localtime(time.time())), write_file_name, confusion_file_name,
                                    seq_confusion_file_name))

                print(word_results_dictionary)
                print(seq_results_dictionary)
                logging.info('Following Evaluation results for features {}'.format(self.model.features_combination))
                logging.info('{}: Evaluation results for chrome number: {} are: \n {} \n {} \n'.
                             format(time.asctime(time.localtime(time.time())), chrome, word_results_dictionary,
                                    seq_results_dictionary))
                logging.info('-----------------------------------------------------------------------------------')

            predict_dict_phase_one = {}
            for sequence_index, seq_word_tag_predict in predict_dict.items():
                predict_dict_phase_one[sequence_index] = seq_word_tag_predict[0]
            return predict_dict_phase_one

        # first phase but without majority approach 2, or second phase of majority approach 2: return the full results
        elif (self.phase_number == 1 and not self.use_majority2) or self.phase_number == 2:
            return predict_dict

    def viterbi_sequence(self, word_tag_list, sequence_index):
        seq_word_tag_predict = {}

        n = len(word_tag_list)
        num_states = len(self.states)

        # create pi and bp numpy
        pi = np.ones(shape=(n+1, num_states, num_states), dtype=float) * float("-inf")
        bp = np.ones(shape=(n+1, num_states, num_states), dtype='int32') * -1

        # initialization: # will be 0 in the numpy
        pi[0, 0, 0] = 1.0
        if self.phase_number == 2:
            word_tag_first_base_predict = self.prediction_for_phase2[sequence_index]
            first_base_predict = word_tag_first_base_predict.split('_')[1]
            pi[1, 0, int(first_base_predict)] = 1.0  # pi[1, #, first_base_predict] = 1 --> freeze the first base
            bp[1, 0, int(first_base_predict)] = 0

        # algorithm:
        # k = phase_number,...,n --> for phase_number=1: start predict from the first base,
        # for phase_number=2: start predict from the second base (second position)
        for k in range(self.phase_number, n+1):
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
            if k in range(1, n-2):
                x_k_p_3 = word_tag_list[k + 2].split('_')[0]  # word k+3
                x_k_p_2 = word_tag_list[k + 1].split('_')[0]  # word k+2
                x_k_p_1 = word_tag_list[k].split('_')[0]      # word k+1
            elif k == n-2:  # word in position n-2, no word in k+3
                x_k_p_3 = '#'  # word k+3
                x_k_p_2 = word_tag_list[k + 1].split('_')[0]  # word k+2
                x_k_p_1 = word_tag_list[k].split('_')[0]      # word k+1
            elif k == n-1:  # word in position n-1, no word in k+3 and k+2
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
                        if self.model_type == 'hmm':  # for HMM calc q*e
                            qe = self.calc_qe(v, u, w, x_k)
                            calc_pi = w_u_pi * qe

                        elif self.model_type == 'memm':  # for MEMM calc q
                            tags_for_matrix = [v, u, w]
                            if '0' in tags_for_matrix:
                                for tag_index, tag in enumerate(tags_for_matrix):
                                    if tag == '0':
                                        tags_for_matrix[tag_index] = '#'
                            # print('memm_v:{}, memm_u:{}, memm_w:{}, x_k_3:{}, x_k_2:{}, x_k_1:{}, x_k_p_3:{},'
                            #       'x_k_p_2:{}, x_k_p_1:{}, x_k:{}'.format(tags_for_matrix[0], tags_for_matrix[1],
                            #                                               tags_for_matrix[2], x_k_3, x_k_2, x_k_1,
                            #                                               x_k_p_3, x_k_p_2, x_k_p_1, x_k))
                            if x_k_p_3 == '' or x_k_p_2 == '' or x_k_p_1 == '' \
                                    or x_k_p_3 == '\n' or x_k_p_2 == '\n' or x_k_p_1 == '\n':
                                print('Error: x_p_i is "" or \n')
                            q = self.calc_q(tags_for_matrix[0], tags_for_matrix[1], tags_for_matrix[2], x_k_3, x_k_2,
                                            x_k_1, x_k_p_3, x_k_p_2, x_k_p_1, x_k)
                            calc_pi = w_u_pi * q

                        else:
                            print('Error: model_type is not in [memm, hmm]')

                        if calc_pi > calc_max_pi:
                            calc_max_pi = calc_pi
                            calc_argmax_pi = int(w)

                    # print int(u), int(v)
                    pi[k, int(u), int(v)] = calc_max_pi  # store the max(pi)
                    bp[k, int(u), int(v)] = calc_argmax_pi  # store the argmax(pi)

        # print pi[n]
        # print bp[n]
        if self.model_type == 'hmm' and self.use_stop_prob:
            stop_p_array = np.ones(shape=(num_states, num_states), dtype=float) * float("-inf")
            x_n_1 = word_tag_list[n - 2].split('_')[0]
            x_n = word_tag_list[n - 1].split('_')[0]
            for u in self.possible_tags(x_n_1):
                for v in self.possible_tags(x_n):
                    u_v_pi = pi[n, int(u), int(v)]
                    transition_stop = self.transition_mat['#' + '|' + u + ',' + v]
                    stop_p = u_v_pi * transition_stop
                    stop_p_array[int(u), int(v)] = stop_p

            u = np.unravel_index(stop_p_array.argmax(), stop_p_array.shape)[0]  # argmax for u in n-1
            v = np.unravel_index(stop_p_array.argmax(), stop_p_array.shape)[1]  # argmax for v in n

            if v in [-1, '-1', 0, '0'] or u in [-1, '-1', 0, '0']:
                print('Error: v or u value is: {}'.format(v))

            seq_word_tag_predict[n - 1] = v
            seq_word_tag_predict[n - 2] = u

            for k in range(n - 2, 0, -1):
                seq_word_tag_predict[k - 1] = bp[k + 2, seq_word_tag_predict[k], seq_word_tag_predict[k + 1]]

            return seq_word_tag_predict

        elif self.model_type == 'memm' or self.use_stop_prob is False:
            u = np.unravel_index(pi[n].argmax(), pi[n].shape)[0]  # argmax for u in n-1
            v = np.unravel_index(pi[n].argmax(), pi[n].shape)[1]  # argmax for v in n

            if v == -1 or u == -1:
                print('Error: v or u value is -1')

            seq_word_tag_predict[n - 1] = v
            seq_word_tag_predict[n - 2] = u

            for k in range(n-2, 0, -1):
                seq_word_tag_predict[k - 1] = bp[k+2, seq_word_tag_predict[k], seq_word_tag_predict[k+1]]
                x_k_m_1 = word_tag_list[k - 1].split('_')[0]
                if (x_k_m_1 == 'A' and seq_word_tag_predict[k - 1] not in ['1' , 1, '5', 5]) or\
                        (x_k_m_1 == 'C' and seq_word_tag_predict[k - 1] not in ['2', 2, '6', 6]) or\
                        (x_k_m_1 == 'G' and seq_word_tag_predict[k - 1] not in ['3', 3, '7', 7]) or\
                        (x_k_m_1 == 'T' and seq_word_tag_predict[k - 1] not in ['4', 4, '8', 8]):
                    reut = 1
            if self.phase_number == 2:
                if int(first_base_predict) != seq_word_tag_predict[0]:
                    print('Error: first base predict is not the final prediction')
            return seq_word_tag_predict

        else:
            print('Error: model_type is not in [memm, hmm]')

    def possible_tags(self, word):
        if word == '#':
            return ['0']
        else:
            # get all relevant tags for word
            return self.word_tag_dict.get(word)

    def calc_qe(self, v, u, w, x_k):  # calculate q*e for HMM model
        tags_for_matrix = [v, u, w]
        for tag_index, tag in enumerate(tags_for_matrix):
            if tag == '0':
                tags_for_matrix[tag_index] = '#'

        q = self.transition_mat[tags_for_matrix[0] + '|' + tags_for_matrix[2] + ',' + tags_for_matrix[1]]
        e = self.emission_mat[x_k + '|' + tags_for_matrix[0]]
        return q * e

    def calc_q(self, v, u, w, x_k_3, x_k_2, x_k_1, x_k_p_3, x_k_p_2, x_k_p_1, x_k):  # calculate q for MEMM model

        sum_denominator = 0
        tag_exp_dict = {}

        for tag in self.word_tag_dict.get(x_k):  # all possible tags for the word x_k
            # history + tag feature vector
            current_history_tag_feature_vector = self.history_tag_feature_vector[(w, u, x_k_3, x_k_2, x_k_1,
                                                                                  x_k_p_1, x_k_p_2, x_k_p_3, x_k), tag]
            # calculate e^(weight*f(history, tag))
            numerators = math.exp(current_history_tag_feature_vector.dot(self.weight))
            sum_denominator += numerators  # sum for the denominator
            tag_exp_dict[tag] = numerators  # save in order to get tag_exp_dict[v]

        return tag_exp_dict[v] / float(sum_denominator)

    def write_majority_doc(self, chrome, dict_seq_results, sequence_index):
        write_file_name = directory + 'majority_vote\\chr' + chrome + '_majority_vote_results_phase'\
                          + str(self.phase_number) + self.model_type + '.csv'
        with open(write_file_name, 'a') as csv_file:
            writer = csv.writer(csv_file)
            if sequence_index == 0:
                writer.writerow(['sequence_index', 'Prediction number (last raw-true label, one before-final prediction)',
                                 'Prediction'])
            for key, value in dict_seq_results.items():
                writer.writerow([key[0], key[1], value])

        return


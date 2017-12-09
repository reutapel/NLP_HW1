import time
import numpy as np
from scipy.sparse import csr_matrix
import csv
from datetime import datetime
import logging
import os.path
from collections import defaultdict


class MEMM:
    """ Base class of modeling MEMM logic on the data"""

#TODO: Add special characters handling
    # shared among all instances of the class'
    STOPS = ['._.','._. "_"']

    def __init__(self, directory, train_file, features_combination, history_tag_feature_vector=False):

        self.tags_dict = {}
        self.word_tag_dict = {}
        self.feature_count = {}
        self.most_common_tags = []
        self.directory = directory
        self.train_file = train_file
        # self.is_create_history_tag_feature_vector = history_tag_feature_vector
        # LOG_FILENAME = datetime.now().strftime(directory + 'logs_MEMM\\LogFileMEMM_%d_%m_%Y_%H_%M.log')
        # logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO)

        self.features_path_string = ''
        for feature in features_combination:
            self.features_path_string += feature + '_'

        self.dict_path = os.path.join(directory + 'dict\\', self.features_path_string)

        # used features
        self.features_combination = features_combination

        self.feature_100 = {}
        self.feature_101 = {}
        self.feature_102 = {}
        self.feature_103 = {}
        self.feature_104 = {}
        self.feature_105 = {}
        self.feature_106 = {}
        self.feature_107 = {}
        self.feature_108 = {}
        self.feature_109 = {}
        self.feature_110 = {}

        # the dictionary that will hold all indexes for all the instances of the features
        self.features_vector = {}

        # mainly for debugging and statistics
        self.features_vector_mapping = {}

        # final vector for Gradient dominator
        self.history_tag_feature_vector_train = defaultdict(list)

        # final vector for Gradient denominator
        self.history_tag_feature_vector_denominator = defaultdict(list)

        # creates history_tag_feature_vector
        if not history_tag_feature_vector:
            # build the type of features
            self.build_features_from_train()
            # build the features_vector
            self.build_features_vector()
            # build the feature vector for all history tuples of seen words and tags
            self.create_history_tag_feature_vector_train()
            # build the feature vector for all history tuples of words in train and their seen tags
            self.create_history_tag_feature_vector_denominator()
        else:
            self.feature_106 = self.read_dict_from_csv('feature_106')
            # self.tags_dict = 
            # self.word_tag_dict = self.read_dict_from_csv('words_tags_dict')
            # self.tags_dict = self.read_dict_from_csv('tags_dict')
            # self.feature_count =
            # self.most_common_tags =
            # self.features_vector = self.read_dict_from_csv('features_vector')
            # self.features_vector_mapping = 2
            # self.history_tag_feature_vector_train =
            # self.history_tag_feature_vector_denominator =


    def read_dict_from_csv(self, dict_name):
        with open(self.dict_path + dict_name + '.csv', mode='rb') as infile:
            reader = csv.reader(infile)
            with open(self.dict_path + dict_name + '.csv', mode='w') as outfile:
                writer = csv.writer(outfile)
                dict = {rows[0]: rows[1] for rows in reader}
        return dict

    def build_features_from_train(self):

        # In this function we are building the features from train data, as well as counting amount of instances from
        # each feature for statistics and feature importance ranking

        start_time = time.time()
        print('{}: starting building features from train'.format(time.asctime(time.localtime(time.time()))))
        logging.info('{}: starting building features from train'.format(time.asctime(time.localtime(time.time()))))

        training_file = self.train_file

        with open(training_file, 'r') as training:
            sequence_index = 1
            for sequence in training:

                sequence = sequence.rstrip('\n')
                word_tag_list = sequence.split(' ')

                # print("working on sequence {} :".format(sequence_index))
                # logging.info("working on sequence {} :".format(sequence_index))
                # print(word_tag_list)
                # logging.info(word_tag_list)

                # define two first and three last word_tags for some features
                first_tag = '*'
                second_tag = '*'
                #zero_word = ''
                #first_word = ''
                second_word = '*'
                plus_one_word = ''
                #plus_two_word = ''
                #plus_three_word = ''

                for word_in_seq_index, word_tag in enumerate(word_tag_list):

                    word_tag_tuple = word_tag.split('_')

                    # TODO: check if "if" is needed and think if junk like ._. before "_" will be a problem
                    # if (word_tag_tuple[0] == '.') & (word_tag_tuple[1] == '.\n'):
                    #     break

                    # # count number of instances for each word in train set
                    # if word_tag_tuple[0] not in self.words_dict:
                    #     self.words_dict[word_tag_tuple[0]] = 1
                    # else:
                    #     self.words_dict[word_tag_tuple[0]] += 1

                    # count number of instances for each tag in train set
                    if word_tag_tuple[1] in self.tags_dict:
                        self.tags_dict[word_tag_tuple[1]] += 1
                    else:
                        self.tags_dict[word_tag_tuple[1]] = 1

                    # count number of all tags seen in train for each word
                    if word_tag_tuple[0] in self.word_tag_dict:
                        if word_tag_tuple[1] in self.word_tag_dict[word_tag_tuple[0]]:
                            self.word_tag_dict[word_tag_tuple[0]][word_tag_tuple[1]] += 1
                        else:
                            self.word_tag_dict[word_tag_tuple[0]][word_tag_tuple[1]] = 1
                        self.word_tag_dict[word_tag_tuple[0]]['COUNT'] += 1
                    else:
                        self.word_tag_dict[word_tag_tuple[0]] = {}
                        self.word_tag_dict[word_tag_tuple[0]]['COUNT'] = 1
                        self.word_tag_dict[word_tag_tuple[0]][word_tag_tuple[1]] = 1

                    current_word = word_tag_tuple[0]
                    current_tag = word_tag_tuple[1]

                    if (word_in_seq_index + 1) == len(word_tag_list):
                        plus_one_word = '*'
                    else:
                        plus_one_word = word_tag_list[word_in_seq_index + 1].split('_')[0]

                    if 'feature_100' in self.features_combination:
                        # build feature_100 of word tag instance
                        feature_100_key = 'f100' + '_' + current_word + '_' + current_tag
                        if feature_100_key in self.feature_100:
                            self.feature_100[feature_100_key] += 1
                        else:
                            self.feature_100[feature_100_key] = 1

                    if 'feature_101' in self.features_combination:
                        # build feature_101 of word suffix and current tag instance
                        if len(current_word) > 3:
                            feature_101_key = 'f101' + '_' + current_word[-3:] + '_' + current_tag
                            if feature_101_key in self.feature_101:
                                self.feature_101[feature_101_key] += 1
                            else:
                                self.feature_101[feature_101_key] = 1

                    if 'feature_102' in self.features_combination:
                        # build feature_102 of word prefix and current tag instance
                        if len(current_word) > 3:
                            feature_102_key = 'f102' + '_' + current_word[:3] + '_' + current_tag
                            if feature_102_key in self.feature_102:
                                self.feature_102[feature_102_key] += 1
                            else:
                                self.feature_102[feature_102_key] = 1

                    if 'feature_103' in self.features_combination:
                        # build feature_103 of tag trigram instance
                        feature_103_key = 'f103' + '_' + first_tag + '_' + second_tag + '_' + current_tag
                        if feature_103_key in self.feature_103:
                            self.feature_103[feature_103_key] += 1
                        else:
                            self.feature_103[feature_103_key] = 1

                    if 'feature_104' in self.features_combination:
                        # build feature_104 of tag bigram instance
                        feature_104_key = 'f104' + '_' + second_tag + '_' + current_tag
                        if feature_104_key in self.feature_104:
                            self.feature_104[feature_104_key] += 1
                        else:
                            self.feature_104[feature_104_key] = 1

                    if 'feature_105' in self.features_combination:
                        # build feature_105 of tag unigram
                        feature_105_key = 'f105' + '_' + current_tag
                        if feature_105_key in self.feature_105:
                            self.feature_105[feature_105_key] += 1
                        else:
                            self.feature_105[feature_105_key] = 1

                    if 'feature_106' in self.features_combination:
                        # build feature_106 of previous word and current tag
                        feature_106_key = 'f106' + '_' + second_word + '_' + current_tag
                        if feature_106_key in self.feature_106:
                            self.feature_106[feature_106_key] += 1
                        else:
                            self.feature_106[feature_106_key] = 1

                    if 'feature_107' in self.features_combination:
                        # build feature_107 of next word and current tag
                        feature_107_key = 'f107' + '_' + plus_one_word + '_' + current_tag
                        if feature_107_key in self.feature_107:
                            self.feature_107[feature_107_key] += 1
                        else:
                            self.feature_107[feature_107_key] = 1

                    # update tags and word
                    first_tag = second_tag
                    second_tag = current_tag
                    second_word = current_word
                sequence_index += 1

        print('{}: finished building features in : {}'.format(time.asctime(time.localtime(time.time())),
                                                            time.time()-start_time))
        logging.info('{}: finished building features in : {}'.format(time.asctime(time.localtime(time.time())),
                                                            time.time()-start_time))

        self.most_common_tags = list(reversed(sorted(self.tags_dict, key=self.tags_dict.get)))

        print('{}: saving dictionaries'.format(time.asctime(time.localtime(time.time()))))
        logging.info('{}: saving dictionaries'.format(time.asctime(time.localtime(time.time()))))

        logging.info('{}: saving tags_dict and most_common_tags'.format(time.asctime(time.localtime(time.time()))))
        w = csv.writer(open(self.dict_path + 'tags_dict' + '.csv', "w"))
        for key, val in self.tags_dict.items():
            w.writerow([key, val])

        w = csv.writer(open(self.dict_path + 'most_common_tags' + '.csv', "w"))
        w.writerow(self.most_common_tags)

        print('{}: finished saving tags_dict and most_common_tags'.format(time.asctime(time.localtime(time.time()))))

        logging.info('saving words_tags_dict')
        w = csv.writer(open( self.dict_path + ' words_tags_dict' + '.csv', "w"))
        for key, val in self.word_tag_dict.items():
            w.writerow([key, val])
        print('{}: finished saving words_tags_dict'.format(time.asctime(time.localtime(time.time()))))

        if 'feature_100' in self.features_combination:
            logging.info('saving feature_100')
            w = csv.writer(open( self.dict_path + 'feature_100' + '.csv', "w"))
            for key, val in self.feature_100.items():
                w.writerow([key, val])
            print('{}: finished saving feature_100'.format(time.asctime(time.localtime(time.time()))))

        if 'feature_101' in self.features_combination:
            logging.info('saving feature_101')
            w = csv.writer(open( self.dict_path + 'feature_101' + '.csv', "w"))
            for key, val in self.feature_101.items():
                w.writerow([key, val])
            print('{}: finished saving feature_101'.format(time.asctime(time.localtime(time.time()))))

        if 'feature_102' in self.features_combination:
            logging.info('saving feature_102')
            w = csv.writer(open( self.dict_path + 'feature_102' + '.csv', "w"))
            for key, val in self.feature_102.items():
                w.writerow([key, val])
            print('{}: finished saving feature_102'.format(time.asctime(time.localtime(time.time()))))

        if 'feature_103' in self.features_combination:
            logging.info('saving feature_103')
            w = csv.writer(open( self.dict_path + 'feature_103' + '.csv', "w"))
            for key, val in self.feature_103.items():
                w.writerow([key, val])
            print('{}: finished saving feature_103'.format(time.asctime(time.localtime(time.time()))))

        if 'feature_104' in self.features_combination:
            logging.info('saving feature_104')
            w = csv.writer(open( self.dict_path + 'feature_104' + '.csv', "w"))
            for key, val in self.feature_104.items():
                w.writerow([key, val])
            print('{}: finished saving feature_104'.format(time.asctime(time.localtime(time.time()))))

        if 'feature_105' in self.features_combination:
            logging.info('saving feature_105')
            w = csv.writer(open( self.dict_path + 'feature_105' + '.csv', "w"))
            for key, val in self.feature_105.items():
                w.writerow([key, val])
            print('{}: finished saving feature_105'.format(time.asctime(time.localtime(time.time()))))

        if 'feature_106' in self.features_combination:
            logging.info('saving feature_106')
            w = csv.writer(open( self.dict_path + 'feature_106' + '.csv', "w"))
            for key, val in self.feature_106.items():
                w.writerow([key, val])
            print('{}: finished saving feature_106'.format(time.asctime(time.localtime(time.time()))))

        if 'feature_107' in self.features_combination:
            logging.info('saving feature_107')
            w = csv.writer(open( self.dict_path + 'feature_107' + '.csv', "w"))
            for key, val in self.feature_107.items():
                w.writerow([key, val])
            print('{}: finished saving feature_107'.format(time.asctime(time.localtime(time.time()))))


        return

    def build_features_vector(self):

        start_time = time.time()
        print('{}: starting building feature vector'.format(time.asctime(time.localtime(time.time()))))
        logging.info('{}: starting building feature vector'.format(time.asctime(time.localtime(time.time()))))

        features_vector_idx = 0
        feature_instances = 0

        if 'feature_100' in self.features_combination:
            # create first type of feature in features_vector which is word tag instances
            for word_tag in self.feature_100.keys():
                self.features_vector[word_tag] = features_vector_idx
                self.features_vector_mapping[features_vector_idx] = word_tag
                features_vector_idx += 1
                feature_instances += 1
            print('{}: size of feature_100 - word+tag instances is: {}'.format(time.asctime(time.localtime(time.time())),
                  feature_instances))
            logging.info('size of feature_100 - word+tag instances is: {}'.
                         format(time.asctime(time.localtime(time.time())), feature_instances))
            feature_instances = 0

        if 'feature_101' in self.features_combination:
            # create second type of feature in features_vector which is word suffix and tag instances
            for word_suffix_tag in self.feature_101.keys():
                self.features_vector[word_suffix_tag] = features_vector_idx
                self.features_vector_mapping[features_vector_idx] = word_suffix_tag
                features_vector_idx += 1
                feature_instances += 1
            print('{}: size of feature_101 - word suffix + tag instances is: {}'.format(time.asctime(time.localtime(time.time())),
                  feature_instances))
            logging.info('size of feature_101 - word suffix + tag instances is: {}'.
                         format(time.asctime(time.localtime(time.time())), feature_instances))
            feature_instances = 0

        if 'feature_102' in self.features_combination:
            # create third type of feature in features_vector which is word prefix tag instances
            for word_prefix_tag in self.feature_102.keys():
                self.features_vector[word_prefix_tag] = features_vector_idx
                self.features_vector_mapping[features_vector_idx] = word_prefix_tag
                features_vector_idx += 1
                feature_instances += 1
            print('{}: size of feature_102 - word prefix +tag instances is: {}'.format(time.asctime(time.localtime(time.time())),
                  feature_instances))
            logging.info('size of feature_102 - word prefix +tag instances is: {}'.
                         format(time.asctime(time.localtime(time.time())), feature_instances))
            feature_instances = 0

        if 'feature_103' in self.features_combination:
            # create forth type of feature in features_vector which is tag trigram instances
            for tags_trigram in self.feature_103.keys():
                self.features_vector[tags_trigram] = features_vector_idx
                self.features_vector_mapping[features_vector_idx] = tags_trigram
                features_vector_idx += 1
                feature_instances += 1
            print('{}: size of feature_103 - tags trigram instances is: {}'.
                  format(time.asctime(time.localtime(time.time())), feature_instances))
            logging.info('{}: size of feature_103 - tags trigram instances is: {}'.
                         format(time.asctime(time.localtime(time.time())), feature_instances))
            feature_instances = 0

        if 'feature_104' in self.features_combination:
            # create fifth type of feature in features_vector which is tag bigram instances
            for tags_bigram in self.feature_104.keys():
                self.features_vector[tags_bigram] = features_vector_idx
                self.features_vector_mapping[features_vector_idx] = tags_bigram
                features_vector_idx += 1
                feature_instances += 1
            print('{}: size of feature_104 - tags bigram is: {}'.format(time.asctime(time.localtime(time.time())),
                                                                        feature_instances))
            logging.info('{}: size of feature_104 - tags bigram is: {}'.format(time.asctime(time.localtime(time.time())),
                                                                           feature_instances))
            feature_instances = 0

        if 'feature_105' in self.features_combination:
            # create sixth type of feature in features_vector which is tag unigram instances
            for tags_unigram in self.feature_105.keys():
                self.features_vector[tags_unigram] = features_vector_idx
                self.features_vector_mapping[features_vector_idx] = tags_unigram
                features_vector_idx += 1
                feature_instances += 1
            print('{}: size of feature_105 - tags unigram is: {}'.format(time.asctime(time.localtime(time.time())),
                                                                        feature_instances))
            logging.info('{}: size of feature_105 - tags unigram is: {}'.format(time.asctime(time.localtime(time.time())),
                                                                           feature_instances))
            feature_instances = 0

        if 'feature_106' in self.features_combination:
            # create seventh type of feature in features_vector which is previous word and current tag instances
            for prev_word_cur_tag in self.feature_106.keys():
                self.features_vector[prev_word_cur_tag] = features_vector_idx
                self.features_vector_mapping[features_vector_idx] = prev_word_cur_tag
                features_vector_idx += 1
                feature_instances += 1
            print('{}: size of feature_106 - previous word and current tag is: {}'.format(time.asctime(time.localtime(time.time())),
                                                                        feature_instances))
            logging.info('{}: size of feature_106 - previous word and current tag is: {}'.format(time.asctime(time.localtime(time.time())),
                                                                           feature_instances))
            feature_instances = 0

        if 'feature_107' in self.features_combination:
            # create eight type of feature in features_vector which is next word and current tag instances
            for next_word_cur_tag in self.feature_107.keys():
                self.features_vector[next_word_cur_tag] = features_vector_idx
                self.features_vector_mapping[features_vector_idx] = next_word_cur_tag
                features_vector_idx += 1
                feature_instances += 1
            print('{}: size of feature_107 - next word and current tag is: {}'.format(
                time.asctime(time.localtime(time.time())),
                feature_instances))
            logging.info('{}: size of feature_107 - next word and current tag is: {}'.format(
                time.asctime(time.localtime(time.time())),
                feature_instances))
            feature_instances = 0

        print('{}: finished building features vector in : {}'.format(time.asctime(time.localtime(time.time())),
              time.time() - start_time))
        logging.info('{}: finished building features vector in : {}'.format(time.asctime(time.localtime(time.time())),
              time.time() - start_time))

        print('{}: saving dictionaries'.format(time.asctime(time.localtime(time.time()))))
        print('{}: saving features_vector'.format(time.asctime(time.localtime(time.time()))))
        logging.info('{}: saving features_vector'.format(time.asctime(time.localtime(time.time()))))
        w = csv.writer(open(self.dict_path + 'features_vector' + '.csv', "w"))
        for key, val in self.features_vector.items():
            w.writerow([key, val])
        print('{}: finished saving features_vector'.format(time.asctime(time.localtime(time.time()))))
        logging.info('{}: finished saving features_vector'.format(time.asctime(time.localtime(time.time()))))

        print('{}: saving features_vector_mapping'.format(time.asctime(time.localtime(time.time()))))
        logging.info('{}: saving features_vector_mapping'.format(time.asctime(time.localtime(time.time()))))
        w = csv.writer(open(self.dict_path + 'features_vector_mapping' + '.csv', "w"))
        for key, val in self.features_vector_mapping.items():
            w.writerow([key, val])
        print('{}: finished saving features_vector_mapping'.format(time.asctime(time.localtime(time.time()))))
        logging.info('{}: finished saving features_vector_mapping'.format(time.asctime(time.localtime(time.time()))))

        return

    def create_history_tag_feature_vector_train(self):

        start_time = time.time()
        print('{}: starting building history_tag_feature_vector_train'.
              format(time.asctime(time.localtime(time.time()))))
        logging.info('{}: starting building history_tag_feature_vector_train'.
                     format(time.asctime(time.localtime(time.time()))))

        training_file = self.train_file
        with open(training_file, 'r') as training:
            sequence_index = 1
            for sequence in training:

                sequence = sequence.rstrip('\n')
                word_tag_list = sequence.split(' ')

                # print("working on sequence {} :".format(sequence_index))
                # print(word_tag_list)
                # define three first word_tags for some features

                first_tag = '*'
                second_tag = '*'
                #zero_word = '*'
                #first_word = '*'
                second_word = '*'
                plus_one_word = ''
                #plus_two_word = ''
                #plus_three_word = ''
                #more_than_3 = True
                last_tuple = True

                for word_in_seq_index, word_tag in enumerate(word_tag_list):

                    word_tag_tuple = word_tag.split('_')

                    # debug last sentence out of range index of last plus one word
                    # if word_in_seq_index == len(word_tag_list)-1:
                    #     print(word_in_seq_index)

                    current_word = word_tag_tuple[0]
                    current_tag = word_tag_tuple[1]

                    if (word_in_seq_index + 1) == len(word_tag_list):
                        plus_one_word = '*'
                    else:
                        plus_one_word = word_tag_list[word_in_seq_index + 1].split('_')[0]

                    # if len(word_tag_list) - word_in_seq_index > 3:
                    #     plus_one_word = word_tag_list[word_in_seq_index + 1].split('_')[0]
                    #     #plus_two_word = word_tag_list[word_in_seq_index + 2][0]
                    #     #plus_three_word = word_tag_list[word_in_seq_index + 3][0]
                    # elif more_than_3:
                    #     # plus_one_word = word_tag_list[word_in_seq_index + 1].split('_')[0]
                    #     # plus_two_word = word_tag_list[word_in_seq_index + 2][0]
                    #     # plus_three_word = '*'
                    #     more_than_3 = False

                    current_history_tag = (first_tag, second_tag, second_word, plus_one_word, current_word), current_tag
                    if current_history_tag in self.history_tag_feature_vector_train:
                        self.history_tag_feature_vector_train[(first_tag, second_tag,
                                                           second_word, plus_one_word, current_word), current_tag][0] += 1
                    else:
                        indexes_vector = self.calculate_history_tag_indexes(first_tag, second_tag, second_word,
                                                                            plus_one_word,
                                                                            current_word, current_tag)
                        self.history_tag_feature_vector_train[(first_tag, second_tag,
                                                           second_word, plus_one_word, current_word), current_tag].append(1)
                        self.history_tag_feature_vector_train[(first_tag, second_tag,
                                                           second_word, plus_one_word, current_word), current_tag].append(indexes_vector)


                    first_tag = second_tag
                    second_tag = current_tag
                    # zero_word = first_word
                    # first_word = second_word
                    second_word = current_word
                    # if not more_than_3:
                    #     plus_one_word = plus_two_word
                    #     plus_two_word = plus_three_word
                sequence_index += 1
        print('{}: finished building history_tag_feature_vector_train in : {}'.
              format(time.asctime(time.localtime(time.time())), time.time() - start_time))
        logging.info('{}: finished building history_tag_feature_vector_train in : {}'.
                     format(time.asctime(time.localtime(time.time())), time.time() - start_time))

        print('{}: saving history_tag_feature_vector_train'.format(time.asctime(time.localtime(time.time()))))
        logging.info('{}: saving history_tag_feature_vector_train'.format(time.asctime(time.localtime(time.time()))))
        w = csv.writer(open(self.dict_path + 'history_tag_feature_vector_train' + '.csv', "w"))
        for key, val in self.history_tag_feature_vector_train.items():
            w.writerow([key, val])
        print('{}: finished saving history_tag_feature_vector_train'.format(time.asctime(time.localtime(time.time()))))
        logging.info('{}: finished saving history_tag_feature_vector_train'.format(time.asctime(time.localtime(time.time()))))

        return

    def create_history_tag_feature_vector_denominator(self):

        start_time = time.time()
        print('{}: starting building history_tag_feature_vector_denominator'.
              format(time.asctime(time.localtime(time.time()))))
        logging.info('{}: starting building history_tag_feature_vector_denominator'.
                     format(time.asctime(time.localtime(time.time()))))

        training_file = self.train_file
        with open(training_file, 'r') as training:
            sequence_index = 1
            for sequence in training:

                sequence = sequence.rstrip('\n')
                word_tag_list = sequence.split(' ')

                # define three first word_tags for some features
                first_tag = '*'
                second_tag = '*'
                #zero_word = '*'
                #first_word = '*'
                second_word = '*'
                plus_one_word = ''
                #plus_two_word = ''
                #plus_three_word = ''
                #more_than_3 = True

                for word_in_seq_index, word_tag in enumerate(word_tag_list):

                    word_tag_tuple = word_tag.split('_')

                    current_word = word_tag_tuple[0]

                    if (word_in_seq_index + 1) == len(word_tag_list):
                        plus_one_word = '*'
                    else:
                        plus_one_word = word_tag_list[word_in_seq_index + 1].split('_')[0]


                    # if len(word_tag_list) - word_in_seq_index > 3:
                    #     plus_one_word = word_tag_list[word_in_seq_index + 1].split('_')[0]
                    #     plus_two_word = word_tag_list[word_in_seq_index + 2][0]
                    #     plus_three_word = word_tag_list[word_in_seq_index + 3][0]
                    # elif more_than_3:
                    #     plus_one_word = word_tag_list[word_in_seq_index + 1].split('_')[0]
                    #     plus_two_word = word_tag_list[word_in_seq_index + 2][0]
                    #     plus_three_word = '*'
                    #     more_than_3 = False

                    for possible_tag_of_current_word in self.word_tag_dict[current_word]:

                        if possible_tag_of_current_word == 'COUNT':
                            continue

                        current_history_tag = (first_tag, second_tag, second_word, plus_one_word,
                                               current_word), possible_tag_of_current_word

                        if current_history_tag in self.history_tag_feature_vector_denominator:
                            self.history_tag_feature_vector_denominator[(first_tag, second_tag,
                                                                   second_word, plus_one_word,
                                                                   current_word), possible_tag_of_current_word][0] += 1
                        else:

                            indexes_vector = self.calculate_history_tag_indexes(first_tag, second_tag, second_word,
                                                                                plus_one_word,
                                                                                current_word,
                                                                                possible_tag_of_current_word)
                            self.history_tag_feature_vector_denominator[(first_tag, second_tag,
                                                                   second_word, plus_one_word,
                                                                   current_word), possible_tag_of_current_word].append(1)
                            self.history_tag_feature_vector_denominator[(first_tag, second_tag,
                                                                   second_word, plus_one_word,
                                                                   current_word), possible_tag_of_current_word].append(indexes_vector)
                    first_tag = second_tag
                    second_tag = word_tag_tuple[1]
                    #zero_word = first_word
                    #first_word = second_word
                    second_word = current_word
                    # if not more_than_3:
                    #     plus_one_word = plus_two_word
                    #     plus_two_word = plus_three_word
                sequence_index += 1

        print('{}: finished building history_tag_feature_vector_denominator in : {}'
              .format(time.asctime(time.localtime(time.time())), time.time() - start_time))
        logging.info('{}: finished building history_tag_feature_vector_denominator in : {}'
                     .format(time.asctime(time.localtime(time.time())), time.time() - start_time))

        print('{}: saving history_tag_feature_vector_denominator'.format(time.asctime(time.localtime(time.time()))))
        logging.info('{}: saving history_tag_feature_vector_denominator'.format(time.asctime(time.localtime(time.time()))))
        w = csv.writer(open(self.dict_path + 'history_tag_feature_vector_denominator' + '.csv', "w"))
        for key, val in self.history_tag_feature_vector_denominator.items():
            w.writerow([key, val])
        print('{}: finished saving history_tag_feature_vector_denominator'.format(time.asctime(time.localtime(time.time()))))
        logging.info('{}: finished saving history_tag_feature_vector_denominator'.format(time.asctime(time.localtime(time.time()))))

        return

    def calculate_history_tag_indexes(self, first_tag, second_tag, second_word, plus_one_word, current_word, current_tag):

        indexes_vector = np.zeros(shape=len(self.features_vector), dtype=int)

        if 'feature_100' in self.features_combination:
            # feature_100 of three tags instances
            feature_100_key = 'f100' + '_' + current_word + '_' + current_tag
            if feature_100_key in self.feature_100:
                feature_idx = self.features_vector[feature_100_key]
                indexes_vector[feature_idx] = 1

        if 'feature_101' in self.features_combination:
            # feature_101 of word suffix and tag instances
            if len(current_word) > 3:
                feature_101_key = 'f101' + '_' + current_word[-3:] + '_' + current_tag
                if feature_101_key in self.feature_101:
                    feature_idx = self.features_vector[feature_101_key]
                    indexes_vector[feature_idx] = 1

        if 'feature_102' in self.features_combination:
            # feature_102 of word prefix and tag instances
            if len(current_word) > 3:
                feature_102_key = 'f102' + '_' + current_word[:3] + '_' + current_tag
                if feature_102_key in self.feature_102:
                    feature_idx = self.features_vector[feature_102_key]
                    indexes_vector[feature_idx] = 1

        if 'feature_103' in self.features_combination:
            # feature_103 of tag trigram instances
            feature_103_key = 'f103' + '_' + first_tag + '_' + second_tag + '_' + current_tag
            if feature_103_key in self.feature_103:
                feature_idx = self.features_vector[feature_103_key]
                indexes_vector[feature_idx] = 1

        if 'feature_104' in self.features_combination:
            # feature_104 of two tags instances
            feature_104_key = 'f104' + '_' + second_tag + '_' + current_tag
            if feature_104_key in self.feature_104:
                feature_idx = self.features_vector[feature_104_key]
                indexes_vector[feature_idx] = 1

        if 'feature_105' in self.features_combination:
            # feature_105 of tag instances
            feature_105_key = 'f105' + '_' + current_tag
            if feature_105_key in self.feature_105:
                feature_idx = self.features_vector[feature_105_key]
                indexes_vector[feature_idx] = 1

        if 'feature_106' in self.features_combination:
            # feature_106 of previous word and current tag instances
            feature_106_key = 'f106' + '_' + second_word + '_' + current_tag
            if feature_106_key in self.feature_106:
                feature_idx = self.features_vector[feature_106_key]
                indexes_vector[feature_idx] = 1

        if 'feature_107' in self.features_combination:
            # feature_107 of next word and current tag instances
            feature_107_key = 'f107' + '_' + plus_one_word + '_' + current_tag
            if feature_107_key in self.feature_107:
                feature_idx = self.features_vector[feature_107_key]
                indexes_vector[feature_idx] = 1

        # efficient representation for sparse vectors that main entrances are zero
        indexes_vector_zip = csr_matrix(indexes_vector)
        return indexes_vector_zip

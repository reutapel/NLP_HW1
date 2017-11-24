import time
from itertools import product
import numpy as np
from scipy.sparse import csr_matrix
import csv

directory = 'C:\\gitprojects\\ML_PROJECT\\'


class MEMM:
    """ Base class of modeling MEMM logic on the data"""

    # shared among all instances of the class'
    amino_mapping = {'TTT' : 'Phe','TTC' : 'Phe','TTA' : 'Leu','TTG' : 'Leu','CTT' : 'Leu','CTC' : 'Leu',
                     'CTA' : 'Leu','CTG' : 'Leu','ATT' : 'Ile','ATC' : 'Ile','ATA' : 'Ile','ATG' : 'Met',
                     'GTT' : 'Val','GTC' : 'Val','GTA' : 'Val','GTG' : 'Val','TCT' : 'Ser','TCC' : 'Ser',
                     'TCA' : 'Ser','TCG' : 'Ser','CCT' : 'Pro','CCC' : 'Pro','CCA' : 'Pro','CCG' : 'Pro',
                     'ACT' : 'Thr','ACC' : 'Thr','ACA' : 'Thr','ACG' : 'Thr','GCT' : 'Ala','GCC' : 'Ala',
                     'GCA' : 'Ala','GCG' : 'Ala','TAT' : 'Tyr','TAC' : 'Tyr','TAA' : 'stop','TAG' : 'stop',
                     'CAT' : 'His','CAC' : 'His','CAA' : 'Gin','CAG' : 'Gin','AAT' : 'Asn','AAC' : 'Asn',
                     'AAA' : 'Lys','AAG' : 'Lys','GAT' : 'Asp','GAC' : 'Asp','GAA' : 'Glu','GAG' : 'Glu',
                     'TGT' : 'Cys','TGC' : 'Cys','TGA' : 'stop','TGG' : 'Trp','CGT' : 'Arg','CGC' : 'Arg',
                     'CGA' : 'Arg','CGG' : 'Arg','AGT' : 'Ser','AGC' : 'Ser','AGA' : 'Arg','AGG' : 'Arg',
                     'GGT' : 'Gly','GGC' : 'Gly','GGA' : 'Gly','GGG' : 'Gly' }

    stop_keys = ['TGA','TAA','TAG']

    start_keys = ['ATG']

    word_tag_dict = {'A': ['1', '5'], 'C': ['2', '6'], 'G': ['3', '7'], 'T': ['4', '8'], '#':['#']}

    def __init__(self, chrome_train_list, features_combination, history_tag_feature_vector=False):

        self.tags_dict = {'1': [0, 'A+'], '2': [0, 'C+'], '3': [0, 'G+'], '4': [0, 'T+'], '5': [0, 'A-'], '6': [0, 'C-'],
                     '7': [0, 'G-'], '8': [0, 'T-']}

        self.words_dict = {'A': 0, 'T': 0, 'C': 0, 'G': 0}

        self.chrome_list = chrome_train_list
        self.is_create_history_tag_feature_vector = history_tag_feature_vector

        # used features
        self.features_combination = features_combination

        self.feature_1 = {}
        self.feature_2 = {}
        self.feature_3 = {}
        self.feature_4 = {}
        self.feature_5 = {}
        self.feature_6 = {}
        self.feature_7 = {}
        self.feature_8 = {}

        # the dictionary that will hold all indexes for all the instances of the features
        self.features_vector = {}

        # mainly for debugging and statistics
        self.features_vector_mapping = {}

        # final vector for Gradient dominator
        self.history_tag_feature_vector_train = {}

        # final vector for Gradient denominator
        self.history_tag_feature_vector_denominator = {}

        # final vector for Viterbi
        self.history_tag_feature_vector = {}

        # build the type of features
        self.build_features_from_train()

        # build the features_vector
        self.build_features_vector()

        # creates history_tag_feature_vector
        if not history_tag_feature_vector:
            self.create_history_tag_feature_vector()
            self.create_history_tag_feature_vector_train()
            self.create_history_tag_feature_vector_denominator()

    def build_features_from_train(self):
        # In this function we are counting amount of instances from
        # each feature for statistics and feature importance

        start_time = time.time()
        print('{}: starting building features from train'.format(time.asctime(time.localtime(time.time()))))

        for chrome in self.chrome_list:
            training_file = directory + 'labels150_non\\chr' + chrome + '_label.csv'

            with open(training_file, 'r') as training:

                sequence_index = 1
                for sequence in training:

                    word_tag_list = sequence.split(',')

                    if '\n' in word_tag_list[len(word_tag_list) - 1]:
                        word_tag_list[len(word_tag_list) - 1] = word_tag_list[len(word_tag_list) - 1].replace('\n',
                                                                                                              '')
                    while '' in word_tag_list:
                        word_tag_list.remove('')
                    while ' ' in word_tag_list:
                        word_tag_list.remove(' ')
                    while '\n' in word_tag_list:
                        word_tag_list.remove('\n')
                    while ',' in word_tag_list:
                        word_tag_list.remove(',')

                    #print("working on sequence {} :".format(sequence_index))
                    #print(word_tag_list)

                    # define two first word_tags for some features
                    first_tag = '#'
                    second_tag = '#'

                    zero_word = ''
                    first_word = ''
                    second_word = ''
                    plus_one_word = ''
                    plus_two_word = ''
                    plus_three_word = ''

                    for word_in_seq_index, word_tag in enumerate(word_tag_list):

                        word_tag_tuple = word_tag.split('_')

                        if '\n' in word_tag_tuple[1]:
                            word_tag_tuple[1] = word_tag_tuple[1][:1]

                        # count number of instances for each word in train set
                        self.words_dict[word_tag_tuple[0]] += 1

                        # count number of instances for each tag in train set
                        self.tags_dict[word_tag_tuple[1]][0] += 1

                        current_word = word_tag_tuple[0]
                        current_tag = word_tag_tuple[1]

                        feature_3_key = ''
                        feature_4_key = ''

                        if 'feature_1' in self.features_combination:
                            # build feature_1 of three tags instances
                            feature_1_key = first_tag + second_tag + current_tag
                            if feature_1_key not in self.feature_1:
                                self.feature_1['f1' + '_' + feature_1_key] = 1
                            else:
                                self.feature_1['f1' + '_' + feature_1_key] += 1

                        if 'feature_2' in self.features_combination:
                            # build feature_2 of two tags instances
                            feature_2_key = second_tag + current_tag
                            if feature_2_key not in self.feature_2:
                                self.feature_2['f2' + '_' + feature_2_key] = 1
                            else:
                                self.feature_2['f2' + '_' + feature_2_key] += 1

                        if word_in_seq_index > 1:
                            first_word = word_tag_list[word_in_seq_index-2][0]
                            second_word = word_tag_list[word_in_seq_index-1][0]
                            feature_3_key = first_word + second_word + current_word
                            feature_4_key = self.amino_mapping[feature_3_key]

                        if 'feature_3' in self.features_combination:
                            # build feature_3 of three words instances
                            if feature_3_key:
                                if feature_3_key not in self.feature_3:
                                    self.feature_3['f3' + '_' + feature_3_key] = 1
                                else:
                                    self.feature_3['f3' + '_' + feature_3_key] += 1

                        if 'feature_4' in self.features_combination:
                            # build feature_4 of amino acids instances
                            if feature_4_key:
                                if feature_4_key not in self.feature_4:
                                    self.feature_4['f4' + '_' + feature_4_key] = 1
                                else:
                                    self.feature_4['f4' + '_' + feature_4_key] += 1

                        if 'feature_5' in self.features_combination:
                            # build feature_5 of stop codon before current word
                            if word_in_seq_index > 2:
                                zero_word = word_tag_list[word_in_seq_index-3][0]
                                # first_word = word_tag_list[word_in_seq_index-2][0]
                                # second_word = word_tag_list[word_in_seq_index-1][0]
                                feature_5_key = zero_word + first_word + second_word
                                if feature_5_key in self.stop_keys:
                                    if feature_5_key not in self.feature_5:
                                        self.feature_5['f5' + '_' + feature_5_key] = 1
                                    else:
                                        self.feature_5['f5' + '_' + feature_5_key] += 1

                        if 'feature_6' in self.features_combination:
                            # build feature_6 of stop codon after current word
                            if len(word_tag_list)-word_in_seq_index > 3:
                                plus_one_word = word_tag_list[word_in_seq_index + 1][0]
                                plus_two_word = word_tag_list[word_in_seq_index + 2][0]
                                plus_three_word = word_tag_list[word_in_seq_index + 3][0]
                                feature_6_key = plus_one_word + plus_two_word + plus_three_word
                                if feature_6_key in self.stop_keys:
                                    if feature_6_key not in self.feature_6:
                                        self.feature_6['f6' + '_' + feature_6_key] = 1
                                    else:
                                        self.feature_6['f6' + '_' + feature_6_key] += 1

                        if 'feature_7' in self.features_combination:
                            # build feature_7 of start codon before current word
                            if word_in_seq_index > 2:
                                # zero_word = word_tag_list[word_in_seq_index - 3][0]
                                # first_word = word_tag_list[word_in_seq_index-2][0]
                                # second_word = word_tag_list[word_in_seq_index-1][0]
                                feature_7_key = zero_word + first_word + second_word
                                if feature_7_key in self.start_keys:
                                    if feature_7_key not in self.feature_7:
                                        self.feature_7['f7' + '_' + feature_7_key] = 1
                                    else:
                                        self.feature_7['f7' + '_' + feature_7_key] += 1

                        if 'feature_8' in self.features_combination:
                            # build feature_8 of start codon after current word
                            if len(word_tag_list) - word_in_seq_index > 3:
                                # plus_one_word = word_tag_list[word_in_seq_index + 1][0]
                                # plus_two_word = word_tag_list[word_in_seq_index + 2][0]
                                # plus_three_word = word_tag_list[word_in_seq_index + 3][0]
                                feature_8_key = plus_one_word + plus_two_word + plus_three_word
                                if feature_8_key in self.start_keys:
                                    if feature_8_key not in self.feature_8:
                                        self.feature_8['f8' + '_' + feature_8_key] = 1
                                    else:
                                        self.feature_8['f8' + '_' + feature_8_key] += 1

                        # update tags
                        first_tag = second_tag
                        second_tag = current_tag
                    sequence_index += 1

        print('{}: finished building features in : {}'.format(time.asctime(time.localtime(time.time())),
                                                              time.time()-start_time))
        return

    def build_features_vector(self):

        start_time = time.time()
        print('{}: starting building feature vector'.format(time.asctime(time.localtime(time.time()))))

        features_vector_idx = 0
        feature_instances = 0

        if 'feature_word_tag' in self.features_combination:
            # create first type of feature in features_vector which is word and tag instances
            for word, tag_list in self.word_tag_dict.items():
                for tag in tag_list:
                    key = word + '_' + tag
                    self.features_vector['wt' + '_' + key] = features_vector_idx
                    self.features_vector_mapping[features_vector_idx] = key
                    features_vector_idx += 1
                    feature_instances += 1
            print('size of feature word and tag instances is: {}'.format(feature_instances))
            feature_instances = 0

        if 'feature_word' in self.features_combination:
            # create second type of feature in features_vector which is instances of words
            for word in self.words_dict.keys():
                self.features_vector['w' + '_' + word] = features_vector_idx
                self.features_vector_mapping[features_vector_idx] = word
                features_vector_idx += 1
                feature_instances += 1
            print('size of feature instances of words is: {}'.format(feature_instances))
            feature_instances = 0

        if 'feature_tag' in self.features_combination:
            # create third type of feature in features_vector which is instances of tags
            for tag in self.tags_dict.keys():
                self.features_vector['t' + '_' + tag] = features_vector_idx
                self.features_vector_mapping[features_vector_idx] = tag
                features_vector_idx += 1
                feature_instances += 1
            print('size of feature instances of tags instances is: {}'.format(feature_instances))
            feature_instances = 0

        if 'feature_1' in self.features_combination:
            # create forth type of feature in features_vector which is three tags instances
            for three_tags in self.feature_1.keys():
                self.features_vector[three_tags] = features_vector_idx
                self.features_vector_mapping[features_vector_idx] = three_tags
                features_vector_idx += 1
                feature_instances += 1
            print('size of feature three tags instances is: {}'.format(feature_instances))
            feature_instances = 0

        if 'feature_2' in self.features_combination:
            # create fifth type of feature in features_vector which is two tags instances
            for two_tags in self.feature_2.keys():
                self.features_vector[two_tags] = features_vector_idx
                self.features_vector_mapping[features_vector_idx] = two_tags
                features_vector_idx += 1
                feature_instances += 1
            print('size of feature two tags instances is: {}'.format(feature_instances))
            feature_instances = 0

        if 'feature_3' in self.features_combination:
            # create sixth type of feature in features_vector which is three words instances
            for three_words in self.feature_3.keys():
                self.features_vector[three_words] = features_vector_idx
                self.features_vector_mapping[features_vector_idx] = three_words
                features_vector_idx += 1
                feature_instances += 1
            print('size of feature three words instances is: {}'.format(feature_instances))
            feature_instances = 0

        if 'feature_4' in self.features_combination:
            # create seventh type of feature in features_vector which is amino instances
            for amino in self.feature_4.keys():
                self.features_vector[amino] = features_vector_idx
                self.features_vector_mapping[features_vector_idx] = amino
                features_vector_idx += 1
                feature_instances += 1
            print('size of feature amino instances is: {}'.format(feature_instances))
            feature_instances = 0

        if 'feature_5' in self.features_combination:
            # create eight type of feature in features_vector which is stop codon before
            for stop_before in self.feature_5.keys():
                self.features_vector[stop_before] = features_vector_idx
                self.features_vector_mapping[features_vector_idx] = stop_before
                features_vector_idx += 1
                feature_instances += 1
            print('size of feature stop codon before is: {}'.format(feature_instances))
            feature_instances = 0

        if 'feature_6' in self.features_combination:
            # create ninth type of feature in features_vector which is stop codon after
            for stop_after in self.feature_6.keys():
                self.features_vector[stop_after] = features_vector_idx
                self.features_vector_mapping[features_vector_idx] = stop_after
                features_vector_idx += 1
                feature_instances += 1
            print('size of feature stop codon after is: {}'.format(feature_instances))
            feature_instances = 0

        if 'feature_7' in self.features_combination:
            # create tenth type of feature in features_vector which is start codon before
            for start_before in self.feature_7.keys():
                self.features_vector[start_before] = features_vector_idx
                self.features_vector_mapping[features_vector_idx] = start_before
                features_vector_idx += 1
                feature_instances += 1
            print('size of feature start codon before is: {}'.format(feature_instances))
            feature_instances = 0

        if 'feature_8' in self.features_combination:
            # create eleventh type of feature in features_vector which is start codon after
            for start_after in self.feature_8.keys():
                self.features_vector[start_after] = features_vector_idx
                self.features_vector_mapping[features_vector_idx] = start_after
                features_vector_idx += 1
                feature_instances += 1
            print('size of feature start codon after is: {}'.format(feature_instances))
            feature_instances = 0

        print('{}: finished building features vector in : {}'.format(time.asctime(time.localtime(time.time())),
              time.time() - start_time))

        return

    def create_history_tag_feature_vector(self):

        start_time = time.time()
        print('{}: starting building history_tag_feature_vector'.format(time.asctime(time.localtime(time.time()))))

        # create all possible keys for feature_vector
        permutations_list = product('ACGT', repeat=7)

        permutations_list_one_t = product('ACGT', repeat=6)
        permutation_list_one = []
        for permutation in permutations_list_one_t:
            permutation_list_one.append(''.join(permutation)+'#')
            permutation_list_one.append('#'+''.join(permutation))
        del(permutations_list_one_t)

        permutations_list_two_t = product('ACGT', repeat=5)
        permutation_list_two = []
        for permutation in permutations_list_two_t:
            permutation_list_two.append(''.join(permutation)+'##')
            permutation_list_two.append('##'+''.join(permutation))
        del(permutations_list_two_t)

        permutations_list_three_t = product('ACGT', repeat=4)
        permutation_list_three = []
        for permutation in permutations_list_three_t:
            permutation_list_three.append(''.join(permutation)+'###')
            permutation_list_three.append('###'+''.join(permutation))
        del(permutations_list_three_t)

        permutation_list_one += permutation_list_two
        permutation_list_one += permutation_list_three
        del(permutation_list_two)
        del (permutation_list_three)
        for permutation in permutations_list:
            permutation_list_one.append(''.join(permutation))
        del(permutations_list)

        for permutation in permutation_list_one:
            word_seq = list(permutation)
            zero_word = word_seq[0]
            first_word = word_seq[1]
            second_word = word_seq[2]
            current_word = word_seq[3]
            plus_one_word = word_seq[4]
            plus_two_word = word_seq[5]
            plus_three_word = word_seq[6]

            possible_tags = [self.word_tag_dict[first_word], self.word_tag_dict[second_word], self.word_tag_dict[current_word]]
            # run on all 8 combinations of possible tags according to given iteration of words
            for possible_tag_comb in list(product(*possible_tags)):

                first_tag = possible_tag_comb[0]
                second_tag = possible_tag_comb[1]
                current_tag = possible_tag_comb[2]
                indexes_vector = self.calculate_history_tag_indexes(first_tag, second_tag,zero_word, first_word, second_word,
                                                                plus_one_word, plus_two_word, plus_three_word,
                                                                    current_word, current_tag)
                self.history_tag_feature_vector[(first_tag, second_tag, zero_word, first_word, second_word,
                                                                plus_one_word, plus_two_word, plus_three_word,
                                                                current_word), current_tag] = indexes_vector

        print('{}: finished building history_tag_feature_vector in : {}'
              .format(time.asctime(time.localtime(time.time())), time.time() - start_time))

        # save history_tag_feature_vector to csv
        # print('writing history_tag_feature_vector to csv')
        # with open('history_tag_feature_vector.csv', 'w') as csv_file:
        #    writer = csv.writer(csv_file)
        #    for key, value in self.history_tag_feature_vector.items():
        #        writer.writerow([key, value])
        # print('FINISHED- writing history_tag_feature_vector to csv')
        return

    def create_history_tag_feature_vector_train(self):

        start_time = time.time()
        print('{}: starting building history_tag_feature_vector_train'.
              format(time.asctime(time.localtime(time.time()))))

        for chrome in self.chrome_list:
            training_file = directory + 'labels150_non\\chr' + chrome + '_label.csv'

            with open(training_file, 'r') as training:

                sequence_index = 1
                for sequence in training:

                    word_tag_list = sequence.split(',')

                    if '\n' in word_tag_list[len(word_tag_list) - 1]:
                        word_tag_list[len(word_tag_list) - 1] = word_tag_list[len(word_tag_list) - 1].replace('\n',
                                                                                                              '')
                    while '' in word_tag_list:
                        word_tag_list.remove('')
                    while ' ' in word_tag_list:
                        word_tag_list.remove(' ')
                    while '\n' in word_tag_list:
                        word_tag_list.remove('\n')
                    while ',' in word_tag_list:
                        word_tag_list.remove(',')

                    # print("working on sequence {} :".format(sequence_index))
                    # print(word_tag_list)

                    # define three first word_tags for some features
                    first_tag = '#'
                    second_tag = '#'

                    zero_word = '#'
                    first_word = '#'
                    second_word = '#'
                    plus_one_word = ''
                    plus_two_word = ''
                    plus_three_word = ''
                    more_than_3 = True

                    for word_in_seq_index, word_tag in enumerate(word_tag_list):

                        word_tag_tuple = word_tag.split('_')

                        if '\n' in word_tag_tuple[1]:
                            word_tag_tuple[1] = word_tag_tuple[1][:1]

                        current_word = word_tag_tuple[0]
                        current_tag = word_tag_tuple[1]
                        if len(word_tag_list) - word_in_seq_index > 3:
                            plus_one_word = word_tag_list[word_in_seq_index + 1][0]
                            plus_two_word = word_tag_list[word_in_seq_index + 2][0]
                            plus_three_word = word_tag_list[word_in_seq_index + 3][0]
                        elif more_than_3:
                            plus_one_word = word_tag_list[word_in_seq_index + 1][0]
                            plus_two_word = word_tag_list[word_in_seq_index + 2][0]
                            plus_three_word = '#'
                            more_than_3 = False

                        indexes_vector = self.calculate_history_tag_indexes(first_tag, second_tag, zero_word,
                                                                            first_word, second_word, plus_one_word,
                                                                            plus_two_word, plus_three_word,
                                                                            current_word, current_tag)
                        self.history_tag_feature_vector_train[(first_tag, second_tag, zero_word, first_word,
                                                               second_word, plus_one_word, plus_two_word,
                                                               plus_three_word, current_word), current_tag] =\
                            indexes_vector
                        first_tag = second_tag
                        second_tag = current_tag
                        zero_word = first_word
                        first_word = second_word
                        second_word = current_word
                        if not more_than_3:
                            plus_one_word = plus_two_word
                            plus_two_word = plus_three_word

                    sequence_index += 1

            # print('finished building history_tag_feature_vector_train in : {}'.format(time.time() - start_time))
            # save history_tag_feature_vector_train to csv
            # print('writing history_tag_feature_vector_train to csv')
            # with open('history_tag_feature_vector_train.csv', 'w') as csv_file:
            #    writer = csv.writer(csv_file)
            #    for key, value in self.history_tag_feature_vector_train.items():
            #        writer.writerow([key, value])
            # print('FINISHED- writing history_tag_feature_vector_train to csv')

        return

    def create_history_tag_feature_vector_denominator(self):

        start_time = time.time()
        print('{}: starting building history_tag_feature_vector_denominator'
              .format(time.asctime(time.localtime(time.time()))))

        for chrome in self.chrome_list:
            training_file = directory + 'labels150_non\\chr' + chrome + '_label.csv'

            with open(training_file, 'r') as training:

                sequence_index = 1
                for sequence in training:

                    word_tag_list = sequence.split(',')

                    if '\n' in word_tag_list[len(word_tag_list) - 1]:
                        word_tag_list[len(word_tag_list) - 1] = word_tag_list[len(word_tag_list) - 1].replace('\n', '')
                    while '' in word_tag_list:
                        word_tag_list.remove('')
                    while ' ' in word_tag_list:
                        word_tag_list.remove(' ')
                    while '\n' in word_tag_list:
                        word_tag_list.remove('\n')
                    while ',' in word_tag_list:
                        word_tag_list.remove(',')

                    # print("working on sequence {} :".format(sequence_index))
                    # print(word_tag_list)

                    # define three first word_tags for some features
                    first_tag = '#'
                    second_tag = '#'

                    zero_word = '#'
                    first_word = '#'
                    second_word = '#'
                    plus_one_word = ''
                    plus_two_word = ''
                    plus_three_word = ''
                    more_than_3 = True

                    for word_in_seq_index, word_tag in enumerate(word_tag_list):

                        word_tag_tuple = word_tag.split('_')

                        if '\n' in word_tag_tuple[1]:
                            word_tag_tuple[1] = word_tag_tuple[1][:1]

                        current_word = word_tag_tuple[0]
                        # current_tag = word_tag_tuple[1]
                        if len(word_tag_list) - word_in_seq_index > 3:
                            plus_one_word = word_tag_list[word_in_seq_index + 1][0]
                            plus_two_word = word_tag_list[word_in_seq_index + 2][0]
                            plus_three_word = word_tag_list[word_in_seq_index + 3][0]
                        elif more_than_3:
                            plus_one_word = word_tag_list[word_in_seq_index + 1][0]
                            plus_two_word = word_tag_list[word_in_seq_index + 2][0]
                            plus_three_word = '#'
                            more_than_3 = False

                        for possible_tag_of_current_word in self.word_tag_dict[current_word]:
                            indexes_vector = self.calculate_history_tag_indexes(first_tag, second_tag, zero_word,
                                                                                first_word, second_word, plus_one_word,
                                                                                plus_two_word, plus_three_word,
                                                                                current_word,
                                                                                possible_tag_of_current_word)
                            self.history_tag_feature_vector_denominator[(first_tag, second_tag, zero_word, first_word,
                                                                         second_word, plus_one_word, plus_two_word,
                                                                         plus_three_word, current_word),
                                                                        possible_tag_of_current_word] = indexes_vector
                        first_tag = second_tag
                        second_tag = word_tag_tuple[1]
                        zero_word = first_word
                        first_word = second_word
                        second_word = current_word
                        if not more_than_3:
                            plus_one_word = plus_two_word
                            plus_two_word = plus_three_word

                    sequence_index += 1

        print('{}: finished building history_tag_feature_vector_denominator in : {}'
              .format(time.asctime(time.localtime(time.time())), time.time() - start_time))
        # save history_tag_feature_vector_train to csv
        # print('writing history_tag_feature_vector_denominator to csv')
        # with open('history_tag_feature_vector_denominator.csv', 'w') as csv_file:
        #    writer = csv.writer(csv_file)
        #    for key, value in self.history_tag_feature_vector_denominator.items():
        #        writer.writerow([key, value])
        # print('FINISHED- writing history_tag_feature_vector_denominator to csv')

        return

    def calculate_history_tag_indexes(self, first_tag, second_tag, zero_word, first_word, second_word,
                                                                plus_one_word, plus_two_word, plus_three_word,
                                                                    current_word, current_tag):
        indexes_vector = np.zeros(shape=len(self.features_vector), dtype=int)

        if 'feature_word_tag' in self.features_combination:
            # first type of feature is word and tag instances
            word_tag = 'wt' + '_' + current_word + '_' + current_tag
            if word_tag in self.features_vector:
                feature_idx = self.features_vector[word_tag]
                indexes_vector[feature_idx] = 1

        if 'feature_word' in self.features_combination:
            # second type of feature is word instances
            if 'w' + '_' + current_word in self.features_vector:
                feature_idx = self.features_vector['w' + '_' + current_word]
                indexes_vector[feature_idx] = 1

        if 'feature_tag' in self.features_combination:
            # third type of feature is tag instances
            if 't' + '_' + current_tag in self.features_vector:
                feature_idx = self.features_vector['t' + '_' + current_tag]
                indexes_vector[feature_idx] = 1

        if 'feature_1' in self.features_combination:
            # feature_1 of three tags instances
            feature_1_key = 'f1' + '_' + first_tag + second_tag + current_tag
            if feature_1_key in self.feature_1:
                feature_idx = self.features_vector[feature_1_key]
                indexes_vector[feature_idx] = 1

        if 'feature_2' in self.features_combination:
            # feature_2 of two tags instances
            feature_2_key = 'f2' + '_' + second_tag + current_tag
            if feature_2_key in self.feature_2:
                feature_idx = self.features_vector[feature_2_key]
                indexes_vector[feature_idx] = 1

        if 'feature_3' in self.features_combination:
            # feature_3 of three words instances
            feature_3_key = 'f3' + '_' + first_word + second_word + current_word
            if feature_3_key in self.feature_3:
                feature_idx = self.features_vector[feature_3_key]
                indexes_vector[feature_idx] = 1

        if 'feature_4' in self.features_combination:
            # feature_4 of amino acids instances
            three_words = first_word + second_word + current_word
            if three_words in self.amino_mapping.keys():
                feature_4_key = 'f4' + '_' + self.amino_mapping[three_words]
                if feature_4_key in self.feature_4:
                    feature_idx = self.features_vector[feature_4_key]
                    indexes_vector[feature_idx] = 1

        if 'feature_5' in self.features_combination:
            # feature_5 of stop codon before current word
            feature_5_key = 'f5' + '_' + zero_word + first_word + second_word
            if feature_5_key in self.feature_5:
                feature_idx = self.features_vector[feature_5_key]
                indexes_vector[feature_idx] = 1

        if 'feature_6' in self.features_combination:
            # feature_6 of stop codon after current word
            feature_6_key = 'f6' + '_' + plus_one_word + plus_two_word + plus_three_word
            if feature_6_key in self.feature_6:
                feature_idx = self.features_vector[feature_6_key]
                indexes_vector[feature_idx] = 1

        if 'feature_7' in self.features_combination:
            # feature_7 of start codon before current word
            feature_7_key = 'f7' + '_' + zero_word + first_word + second_word
            if feature_7_key in self.feature_7:
                feature_idx = self.features_vector[feature_7_key]
                indexes_vector[feature_idx] = 1

        if 'feature_8' in self.features_combination:
            # feature_8 of start codon after current word
            feature_8_key = 'f8' + '_' + plus_one_word + plus_two_word + plus_three_word
            if feature_8_key in self.feature_8:
                feature_idx = self.features_vector[feature_8_key]
                indexes_vector[feature_idx] = 1

        # efficient representation
        if self.is_create_history_tag_feature_vector:  # non structure classifier
            return indexes_vector
        else:  # memm
            indexes_vector_zip = csr_matrix(indexes_vector)
            return indexes_vector_zip

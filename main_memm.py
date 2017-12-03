from MEMM_try import MEMM
from viterbi_ML import viterbi
import time
from Print_and_save_results import print_save_results
import numpy as np
from gradient_try import Gradient
import logging
from datetime import datetime
import itertools
import sys

directory = 'C:\\Users\\Meir\\PycharmProjects\\ML_PROJECT\\'
LOG_FILENAME = datetime.now().strftime(directory + 'logs\\LogFileMEMM_%d_%m_%Y_%H_%M.log')
logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO)

def main():
    # chrome_train_list = ['1', '2', '3', '4','14', '6', '7', '8', '9', '10', '11', '12','17','15']
    # chrome_test_list = ['13','5', '16',]
    # features_combination_list_sub = ['feature_word_tag','feature_word','feature_tag','feature_1','feature_2',
    #                              'feature_3','feature_4','feature_5','feature_6','feature_7',
    #                              'feature_8']

    features_combination_list = [['feature_word_tag', 'feature_word', 'feature_tag'],
                                 ['feature_word_tag', 'feature_word', 'feature_tag', 'feature_1', 'feature_2',
                                  'feature_3', 'feature_4']]

    #for perm in itertools.combinations(features_combination_list_sub, 5):
    #    features_combination_list.append(list(perm))
    #for perm in itertools.combinations(features_combination_list_sub, 6):
    #    features_combination_list.append(list(perm))
    # for perm in itertools.combinations(features_combination_list_sub, 7):
    #     features_combination_list.append(list(perm))
    # for perm in itertools.combinations(features_combination_list_sub, 8):
    #     features_combination_list.append(list(perm))
    # for perm in itertools.combinations(features_combination_list_sub, 9):
    #     features_combination_list.append(list(perm))
    # for perm in itertools.combinations(features_combination_list_sub, 10):
    #     features_combination_list.append(list(perm))
    #for perm in itertools.combinations(features_combination_list_sub, 2):
    #    features_combination_list.append(list(perm))
    #for perm in itertools.combinations(features_combination_list_sub, 3):
    #    features_combination_list.append(list(perm))
    #for perm in itertools.combinations(features_combination_list_sub, 4):
    #    features_combination_list.append(list(perm))

    directory = 'C:\\Users\\ssheiba\\Desktop\\MASTER\\NLP\\HW1\\NLP_HW1\\data'

    logging.info('{}: Train list is: {}, test list is: {}'
                 .format(time.asctime(time.localtime(time.time())), chrome_train_list, chrome_test_list))
    print('{}: Start creating MEMM'.format(time.asctime(time.localtime(time.time()))))
    logging.info('{}: Start creating MEMM'.format(time.asctime(time.localtime(time.time()))))
    for features_combination in features_combination_list:

        logging.info('MEMM for features : {}'.format(features_combination))

        memm_class = MEMM(directory, chrome_train_list, features_combination)

        gradient_class = Gradient(memm=memm_class, lamda=1)
        gradient_result = gradient_class.gradient_descent()
        weights = gradient_result.x
        #np.savetxt(gradient_file, weights, delimiter=",")

        for chrome in chrome_test_list:
            if chrome == '3' and features_combination == ['feature_word_tag', 'feature_word', 'feature_tag']:
                continue
            test_file = directory + 'labels150\\chr' + chrome + '_label.csv'
            print('{}: Start viterbi for chrome: {}'.format((time.asctime(time.localtime(time.time()))), chrome))
            viterbi_class = viterbi(memm_class, 'memm', data_file=test_file, is_log=False, use_stop_prob=False, w=weights)
            viterbi_result = viterbi_class.viterbi_all_data(chrome)

            write_file_name = datetime.now().strftime\
                (directory + 'file_results\\chr' + chrome + '_resultMEMM_%d_%m_%Y_%H_%M.csv')
            confusion_file_name = datetime.now().strftime\
                (directory + 'confusion_files\\chr' + chrome + '_CMMEMM_%d_%m_%Y_%H_%M.xls')
            seq_confusion_file_name = datetime.now().strftime\
                (directory + 'confusion_files\\chr' + chrome + '_sqeCMMEMM_%d_%m_%Y_%H_%M.xls')
            # seq_labels_file_name = 'C:/gitprojects/ML project/samples_small_data/chr1_sample_label.xlsx'
            seq_labels_file_name = directory + 'sample_labels150\\chr' + chrome + '_sample_label.xlsx'
            evaluate_class = print_save_results(memm_class, 'memm', test_file, viterbi_result, write_file_name,
                                              confusion_file_name, seq_labels_file_name, seq_confusion_file_name)
            word_results_dictionary, seq_results_dictionary = evaluate_class.run()

            logging.info('{}: Related results files are: \n {} \n {} \n {}'.
                         format(time.asctime(time.localtime(time.time())), write_file_name, confusion_file_name,
                                seq_confusion_file_name))

            print(word_results_dictionary)
            print(seq_results_dictionary)
            logging.info('Following Evaluation results for features {}'.format(features_combination))
            logging.info('{}: Evaluation results for chrome number: {} are: \n {} \n {} \n'.
                         format(time.asctime(time.localtime(time.time())), chrome, word_results_dictionary,
                                seq_results_dictionary))
            logging.info('-----------------------------------------------------------------------------------')

if __name__ == "__main__":
    all_chromes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17']
    for test_chrome in range(3, 18):
        chrome_train_list = [x for x in all_chromes if x != str(test_chrome)]
        print(chrome_train_list)
        chrome_test_list = [str(test_chrome)]
        print(chrome_test_list)
        main()

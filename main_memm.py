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


def main():

    # open log connection
    directory = 'C:\\Users\\ssheiba\\Desktop\\MASTER\\NLP\\HW1\\NLP_HW1\\'
    LOG_FILENAME = datetime.now().strftime(directory + 'logs_MEMM\\LogFileMEMM_MAIN_%d_%m_%Y_%H_%M.log')
    logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO)
    print('{}: Start creating MEMM'.format(time.asctime(time.localtime(time.time()))))
    logging.info('{}: Start creating MEMM'.format(time.asctime(time.localtime(time.time()))))


    #for perm in itertools.combinations(features_combination_list_sub, 4):
    #    features_combination_list.append(list(perm))


    # start all combination of features
    for features_combination in features_combination_list:

        logging.info('started MEMM for features : {}'.format(features_combination))
        print('started MEMM for features : {}'.format(features_combination))

        memm_class = MEMM(directory, features_combination)

        logging.info('finished MEMM for features : {}'.format(features_combination))
        print('finished MEMM for features : {}'.format(features_combination))

        print('started gradient for features : {}'.format(features_combination))
        logging.info('started gradient for features : {}'.format(features_combination))

        gradient_class = Gradient(memm=memm_class, lamda=1)
        gradient_result = gradient_class.gradient_descent()

        print('finished gradient for features : {}'.format(features_combination))
        logging.info('finished gradient for features : {}'.format(features_combination))

        weights = gradient_result.x
        #np.savetxt(gradient_file, weights, delimiter=",")

        test_file = directory + 'data\\test.wtag'

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

    all_features = ['feature_100', 'feature_101', 'feature_102', 'feature_103', 'feature_104', 'feature_105',
                    'feature_106', 'feature_107', 'feature_108', 'feature_109', 'feature_110']
    basic_model = [['feature_100', 'feature_103', 'feature_104']]

    features_combination_list = basic_model
    main()

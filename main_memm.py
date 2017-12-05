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

# open log connection
directory = '/Users/reutapel/Documents/Technion/Msc/NLP/hw1/NLP_HW1/'
LOG_FILENAME = datetime.now().strftime(directory + 'logs_MEMM/LogFileMEMM_MAIN_%d_%m_%Y_%H_%M.log')
logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO)


def main(train_file_to_use, test_file_to_use, test_type, features_combination_list):

    print('{}: Start creating MEMM'.format(time.asctime(time.localtime(time.time()))))
    logging.info('{}: Start creating MEMM'.format(time.asctime(time.localtime(time.time()))))
    #for perm in itertools.combinations(features_combination_list_sub, 4):
    #    features_combination_list.append(list(perm))

    # start all combination of features
    for features_combination in features_combination_list:

        logging.info('started MEMM for features : {}'.format(features_combination))
        print('started MEMM for features : {}'.format(features_combination))

        memm_class = MEMM(directory, train_file_to_use, features_combination)

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

        print('{}: Start viterbi'.format((time.asctime(time.localtime(time.time())))))
        viterbi_class = viterbi(memm_class, data_file=test_file_to_use, w=weights)
        viterbi_result = viterbi_class.viterbi_all_data()

        write_file_name = datetime.now().strftime(directory + 'file_results\\result_MEMM_' + test_type +
                                                  '%d_%m_%Y_%H_%M.csv')
        confusion_file_name = datetime.now().strftime(directory + 'confusion_files\\CM_MEMM_' + test_type +
                              '%d_%m_%Y_%H_%M.xls')

        evaluate_class = print_save_results(memm_class, test_file_to_use, viterbi_result, write_file_name,
                                            confusion_file_name)
        word_results_dictionary = evaluate_class.run()

        logging.info('{}: Related results files are: \n {} \n {}'.
                     format(time.asctime(time.localtime(time.time())), write_file_name, confusion_file_name))

        print(word_results_dictionary)
        logging.info('Following Evaluation results for features {}'.format(features_combination))
        logging.info('{}: Evaluation results are: \n {} \n'.format(time.asctime(time.localtime(time.time())),
                                                                   word_results_dictionary))
        logging.info('-----------------------------------------------------------------------------------')


if __name__ == "__main__":
    train_file = directory + 'data/train.wtag'
    test_file = directory + 'data/test.wtag'
    comp_file = directory + 'data/comp.words'

    feature_type_dict = {'all_features': [['feature_100', 'feature_101', 'feature_102', 'feature_103', 'feature_104',
                                          'feature_105', 'feature_106', 'feature_107', 'feature_108', 'feature_109',
                                           'feature_110']],
                         'basic_model': [['feature_100', 'feature_103', 'feature_104']]}

    for feature_type_name, feature_type_list in feature_type_dict.items():
        main(train_file, test_file, 'test', feature_type_list)

from MEMM_try import MEMM
from viterbi_ML import viterbi
import time
from Print_and_save_results import print_save_results
from gradient_try import Gradient
import logging
from datetime import datetime

directory = '/Users/reutapel/Documents/Technion/Msc/NLP/hw1/NLP_HW1/'
LOG_FILENAME = datetime.now().strftime(directory + 'logs\\LogFileMEMM_%d_%m_%Y_%H_%M.log')
logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO)


def main(test_file_to_use, test_type):
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

    print('{}: Start creating MEMM'.format(time.asctime(time.localtime(time.time()))))
    logging.info('{}: Start creating MEMM'.format(time.asctime(time.localtime(time.time()))))
    for features_combination in features_combination_list:

        logging.info('MEMM for features : {}'.format(features_combination))

        memm_class = MEMM(train_file, features_combination)

        gradient_class = Gradient(memm=memm_class, lamda=1)
        gradient_result = gradient_class.gradient_descent()
        weights = gradient_result.x
        #np.savetxt(gradient_file, weights, delimiter=",")

        print('{}: Start viterbi'.format((time.asctime(time.localtime(time.time())))))
        viterbi_class = viterbi(memm_class, data_file=test_file_to_use, use_stop_prob=False, w=weights)
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
    main(test_file, 'test')

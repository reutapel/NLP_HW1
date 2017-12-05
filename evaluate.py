import xlwt
import itertools
import pandas as pd


class Evaluate:
    """
    this class evaluates the results and creates the confusion matrix
    """
    # todo: print the Viterbi results - to see the difference from the test
    # todo: calculate the misses and hits of tags and words
    # todo: calculate the accuracy of the model
    # todo: create the confusion matrix - and create the file for it
    # todo: evaluate and present the worst 10 tags errors

    def __init__(self, model, data_file, viterbi_result, write_file_name,
                 confusion_file_name):
        """

        :param model: MEMM model object
        :param data_file: test data file
        :param viterbi_result: results of the viterbi results
        :type viterbi_result: dict with key:= sentence_index, value:= list (in form word_tag)
        :param write_file_name: where to save the Viterbi results
        :param confusion_file_name: where to save the confusion matrix
        """
        self.data_file_name = data_file
        self.viterbi_result = viterbi_result
        self.model = model
        self.write_file_name = write_file_name
        self.confusion_file_name = confusion_file_name
        # self.seq_confusion_file_name = seq_confusion_file_name
        self.tags = list(itertools.chain.from_iterable(model.tags_dict))
        self.tags.remove('COUNT')
        self.tags.sort()
        # self.word_tag_dict = model.word_tag_dict
        # seq_label = pd.read_excel(seq_labels_file_name, header=None)
        # self.seq_label = seq_label.as_matrix()

        self.confusion_matrix = {}
        self.all_seq_confusion_matrix = {}
        self.eval_res = {}
        self.word_results_dictionary, self.seq_results_dictionary = self.eval_test_results(self.viterbi_result,
                                                                                           self.data_file_name)

    def run(self):
        self.write_result_doc()
        self.write_confusion_doc()  # write tags confusion matrix

        return self.word_results_dictionary, self.seq_results_dictionary

    def eval_test_results(self, predicted_word_tag, data_file_name):
        # print('predicted_word_tag is: {}').format(predicted_word_tag)
        # predicted_values
        miss = 0
        hit = 0

        for tag1 in self.tags:
            for tag2 in self.tags:
                tag_key = tag1 + '_' + tag2
                self.confusion_matrix.setdefault(tag_key, 0)

        word_tag_tuples_dict = {}
        # with open(data_file_name, 'r') as training:  # real values
        #     for sequence in training:
        # todo: consider make the test tagging one time
        with open(data_file_name, 'r') as train:
            for index, seq in enumerate(train):
                seq = seq.rstrip('\n')
                d = seq.split(' ')
                word_tag_tuples_dict[index] = []
                for i, val in enumerate(d):
                    word_tag_tuples_dict[index].append(val.split('_'))
                    predict_tuple = predicted_word_tag[index][i].split('_')
                    # print('sequence_index is: {}, predict_item is: {}').format(sequence_index, predict_item)
                    predict_word = predict_tuple[0]
                    predict_tag = predict_tuple[1]  # our predicted tag
                    test_word = word_tag_tuples_dict[index][i][0]
                    test_tag = word_tag_tuples_dict[index][i][1]
                    if predict_word != test_word:
                        print('problem miss between prediction word: {0} and test word {1} indexes : {2}'
                              .format(predict_word, test_word, str((i, index))))
                    if predict_tag != test_tag:  # tag miss
                        miss += 1
                        confusion_matrix_key = str(test_tag) + '_' + str(predict_tag)  # real tag _ prediction tag
                        self.confusion_matrix[confusion_matrix_key] += 1

                    else:
                        hit += 1
                        confusion_mat_key = str(test_tag) + '_' + str(predict_tag)  # trace add
                        self.confusion_matrix[confusion_mat_key] += 1

        print('Misses: {0}, Hits: {1}'.format(miss, hit))
        print('Model Accuracy: {0}'.format(float(hit)/float(miss+hit)))
        # print('Miss per word')
        # print(miss)
        # print('Hit per word')
        # print(hit)
        # print('Accuracy per word')
        # print(float(hit)/float(miss+hit))

        return \
            {
                'Miss per word': miss,
                'Hit per word': hit,
                'Accuracy per word': float(hit)/float(miss+hit),
                'confusion_matrix per word': self.confusion_matrix
             }

    def write_result_doc(self):

        file_name = self.write_file_name
        len(self.viterbi_result)
        with open(file_name, 'w') as f:
            for sentence_index, sequence_list in self.viterbi_result.items():
                for idx_inner, word_tag_string in enumerate(sequence_list):
                    f.write("{0} ".format(word_tag_string))
                f.write('\n')                                           # finish sentences

        return

    def write_confusion_doc(self):
        """
            build confusion matrix doc
            build structure of line and columns
        """

        file_name = self.confusion_file_name
        column_rows_structure = self.tags
        confusion_matrix_to_write = self.confusion_matrix

        book = xlwt.Workbook(encoding="utf-8")

        sheet1 = book.add_sheet("Confusion Matrix")

        pattern = xlwt.Pattern()
        pattern.pattern = xlwt.Pattern.SOLID_PATTERN
        pattern.pattern_fore_colour = 22
        style = xlwt.XFStyle()
        style.pattern = pattern

        pattern_mistake = xlwt.Pattern()
        pattern_mistake.pattern = xlwt.Pattern.SOLID_PATTERN
        pattern_mistake.pattern_fore_colour = 2
        style_mistake = xlwt.XFStyle()
        style_mistake.pattern = pattern_mistake

        pattern_good = xlwt.Pattern()
        pattern_good.pattern = xlwt.Pattern.SOLID_PATTERN
        pattern_good.pattern_fore_colour = 3
        style_good = xlwt.XFStyle()
        style_good.pattern = pattern_good

        sheet1.write(0, 0, ' ', style)
        for idx_tag, cur_tag in enumerate(column_rows_structure):
            sheet1.write(0, idx_tag+1, cur_tag, style)

        for row_tag_idx, row_tag in enumerate(column_rows_structure):
            sheet1.write(row_tag_idx+1, 0, row_tag, style)
            for col_tag_idx, col_tag in enumerate(column_rows_structure):
                cur_value = confusion_matrix_to_write[str(row_tag) + '_' + str(col_tag)]
                if cur_value == 0:
                    sheet1.write(row_tag_idx + 1, col_tag_idx+1, str(cur_value))
                else:
                    if row_tag_idx == col_tag_idx:
                        sheet1.write(row_tag_idx + 1, col_tag_idx + 1, str(cur_value), style_good)
                    else:
                        sheet1.write(row_tag_idx + 1, col_tag_idx + 1, str(cur_value), style_mistake)
        book.save(file_name)

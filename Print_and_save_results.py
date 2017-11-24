import xlwt
import itertools
import pandas as pd


class print_save_results:

    def __init__(self, model, model_type, data_file, viterbi_result, write_file_name,
                 confusion_file_name, seq_labels_file_name, seq_confusion_file_name):
        # model will be HMM or MEMM object, model_type in ['hmm','memm']
        self.data_file_name = data_file
        self.model_type = model_type
        self.viterbi_result = viterbi_result
        self.model = model
        self.write_file_name = write_file_name
        self.confusion_file_name = confusion_file_name
        self.seq_confusion_file_name = seq_confusion_file_name
        self.states = list(itertools.chain.from_iterable(model.word_tag_dict.values()))
        self.states.remove('#')
        self.states.sort()
        # self.word_tag_dict = model.word_tag_dict
        seq_label = pd.read_excel(seq_labels_file_name, header=None)
        self.seq_label = seq_label.as_matrix()

        self.confusion_matrix = {}
        self.all_seq_confusion_matrix = {}
        self.eval_res = {}

    def run(self):
        self.word_results_dictionary, self.seq_results_dictionary = self.eval_test_results(self.viterbi_result,
                                                                                           self.data_file_name)
        self.write_result_doc()
        self.write_confusion_doc(True)  # write tags confusion matrix
        self.write_confusion_doc(False)  # write sequence label confusion matrix

        return self.word_results_dictionary, self.seq_results_dictionary

    def eval_test_results(self, predicted_word_tag, data_file_name):
        # print('predicted_word_tag is: {}').format(predicted_word_tag)
        # predicted_values
        miss = 0
        hit = 0
        seq_miss = 0
        seq_hit = 0

        for tag1 in self.states:
            for tag2 in self.states:
                cur_key = tag1 + '_' + tag2
                if cur_key not in self.confusion_matrix:
                    self.confusion_matrix[cur_key] = 0

        for seq_tag1 in ['1', '-1']:
            for seq_tag2 in ['1', '-1']:
                cur_key = seq_tag1 + '_' + seq_tag2
                if cur_key not in self.all_seq_confusion_matrix:
                    self.all_seq_confusion_matrix[cur_key] = 0

        sequence_index = 0
        with open(data_file_name, 'r') as training:  # real values
            for sequence in training:
                include_gen = -1
                word_tag_list = sequence.split(',')
                while ' ' in word_tag_list:
                    word_tag_list.remove(' ')
                while '' in word_tag_list:
                    word_tag_list.remove('')
                while '\n' in word_tag_list:
                    word_tag_list.remove('\n')
                for i, val in enumerate(word_tag_list):
                    word_tag_tuple = val.split('_')
                    if '\n' in word_tag_tuple[1]:  # end of sequence
                        word_tag_tuple[1] = word_tag_tuple[1][:1]
                    # if '\n' in word_tag_tuple[1]:
                    #     word_tag_tuple[1] = word_tag_tuple[1][:1]
                    predict_item = predicted_word_tag[sequence_index][i].split('_')
                    # print('sequence_index is: {}, predict_item is: {}').format(sequence_index, predict_item)
                    predict_word = predict_item[0]
                    predict_tag = predict_item[1]  # our predicted tag
                    if include_gen == -1 and predict_tag in ['1', '2', '3', '4']:
                        include_gen = 1
                    if predict_word != word_tag_tuple[0]:
                        print('problem miss between prediction and test word indexes')
                    if predict_tag != word_tag_tuple[1]:  # tag miss
                        miss += 1
                        confusion_mat_key = str(word_tag_tuple[1]) + '_' + str(predict_tag)  # real tag _ prediction tag
                        self.confusion_matrix[confusion_mat_key] += 1

                    else:
                        hit += 1
                        confusion_mat_key = str(word_tag_tuple[1]) + '_' + str(predict_tag)  # trace add
                        self.confusion_matrix[confusion_mat_key] += 1

                true_include_gen = self.seq_label[sequence_index][0]
                if include_gen != true_include_gen:
                    seq_miss += 1
                    seq_confusion_mat_key = str(true_include_gen) + '_' + str(include_gen)  # real tag _ prediction tag
                    self.all_seq_confusion_matrix[seq_confusion_mat_key] += 1

                else:
                    seq_hit += 1
                    seq_confusion_mat_key = str(true_include_gen) + '_' + str(include_gen)  # real tag _ prediction tag
                    self.all_seq_confusion_matrix[seq_confusion_mat_key] += 1
                sequence_index += 1

        print('Miss')
        print(miss)
        print('Hit')
        print(hit)
        print('Accuracy')
        print(float(hit)/float(miss+hit))
        print('Miss per word')
        print(miss)
        print('Hit per word')
        print(hit)
        print('Accuracy per word')
        print(float(hit)/float(miss+hit))
        print('Miss per seq')
        print(seq_miss)
        print('Hit per seq')
        print(seq_hit)
        print('Accuracy per seq')
        print(float(seq_hit)/float(seq_miss+seq_hit))

        return \
            {
                'Miss per word': miss,
                'Hit per word': hit,
                'Accuracy per word': float(hit)/float(miss+hit),
                'confusion_matrix per word': self.confusion_matrix
             }, \
            {
                'Miss per seq': seq_miss,
                'Hit per seq': seq_hit,
                'Accuracy per seq': float(seq_hit)/float(seq_miss+seq_hit),
                'confusion_matrix per seq': self.all_seq_confusion_matrix
             }

    def write_result_doc(self):

        file_name = self.write_file_name
        f = open(file_name, 'w')

        for sequence_index, sequence_list in self.viterbi_result.items():
            for idx_inner, word_tag_string in enumerate(sequence_list):
                f.write(word_tag_string+',')
            f.write('\n')                                           # finish sentences
        f.close()

        return

    def write_confusion_doc(self, is_tags_conf):
        # build confusion matrix doc
        # build structure of line and columns

        if is_tags_conf:
            file_name = self.confusion_file_name
            column_rows_structure = self.states
            confusion_matrix_to_write = self.confusion_matrix
        else:
            file_name = self.seq_confusion_file_name
            column_rows_structure = ['1', '-1']
            confusion_matrix_to_write = self.all_seq_confusion_matrix

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

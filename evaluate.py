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
        self.predict_dict, self.viterbi_unseen_words = viterbi_result
        self.viterbi_unseen_words = [list(cord) for cord in set(tuple(cord) for cord in self.viterbi_unseen_words)]
        self.model = model
        self.write_file_name = write_file_name
        self.confusion_file_name = confusion_file_name
        self.tags = list(model.tags_dict.keys())
        self.tags.sort()
        self.unseen_confusion_matrix = {}
        self.confusion_matrix = {}
        self.misses_matrix = {}
        self.eval_res = {}
        self.k = 10  # number of words in the confusion matrix
        self.unseen_tags_set = set()
        self.word_results_dictionary = self.eval_test_results()

    def run(self):
        self.write_result_doc()
        self.write_confusion_doc()  # write tags confusion matrix

        return self.word_results_dictionary

    def eval_test_results(self):
        # predicted_values
        miss = 0
        hit = 0
        hit_unseen = 0
        miss_unseen = 0

        for tag1 in self.tags:
            for tag2 in self.tags:
                tag_key = tag1 + '_' + tag2
                self.confusion_matrix.setdefault(tag_key, 0)
                self.misses_matrix.setdefault(tag_key, 0)

        word_tag_tuples_dict = []
        # with open(data_file_name, 'r') as training:  # real values
        #     for sequence in training:
        # todo: consider make the test tagging one time
        with open(self.data_file_name, 'r') as train:
            for index, seq in enumerate(train):
                seq = seq.rstrip('\n')
                d = seq.split(' ')
                word_tag_tuples_dict.append([])
                for i, val in enumerate(d):
                    word_tag_tuples_dict[index].append(val.split('_'))
                    predict_tuple = self.predict_dict[index][i].split('_')
                    # print('sequence_index is: {}, predict_item is: {}').format(sequence_index, predict_item)
                    predict_word = predict_tuple[0]
                    predict_tag = predict_tuple[1]  # our predicted tag
                    gold_word = word_tag_tuples_dict[index][i][0]
                    gold_tag = word_tag_tuples_dict[index][i][1]
                    if predict_word != gold_word:
                        print('problem between prediction word: {0} and test word {1} indexes : {2}'
                              .format(predict_word, gold_word, str((index, i))))
                    confusion_matrix_key = "{0}_{1}".format(gold_tag, predict_tag)
                    if gold_tag not in self.tags:
                        self.add_missing_tags(gold_tag, predict_tag)
                    self.confusion_matrix[confusion_matrix_key] += 1
                    if predict_tag != gold_tag:  # tag miss
                        miss += 1
                        self.misses_matrix[confusion_matrix_key] += 1
                    else:
                        hit += 1

        print('Misses: {0}, Hits: {1}'.format(miss, hit))
        print('Model Accuracy: {0}'.format(float(hit)/float(miss+hit)))
        
        for unseen_word in self.viterbi_unseen_words:
            sentence_idx, word_idx = unseen_word
            gold_word, gold_tag = word_tag_tuples_dict[sentence_idx][word_idx]
            predict_word, predict_tag = self.predict_dict[sentence_idx][word_idx].split('_')
            if gold_word != predict_word:
                print('problem between prediction word: {0} and test word {1} indexes : {2}'
                      .format(predict_word, gold_word, str((sentence_idx, word_idx))))
            self.unseen_tags_set.update((gold_tag, predict_tag))
            keys = self.get_all_possible_tags(gold_tag, predict_tag)
            for key in keys:
                self.unseen_confusion_matrix.setdefault(key, 0)
            confusion_matrix_key = "{0}_{1}".format(gold_tag, predict_tag)
            self.unseen_confusion_matrix[confusion_matrix_key] += 1
            if predict_tag != gold_tag:
                miss_unseen += 1
            else:
                hit_unseen += 1
        print('Unseen Confusion')
        print('Misses: {0}, Hits: {1}'.format(miss_unseen, hit_unseen))
        print('Model Accuracy: {0}'.format(float(hit_unseen) / float(miss_unseen + hit_unseen)))

        unseen_tag_list = sorted(self.unseen_tags_set)

        keys_set = set()
        for i in range(len(unseen_tag_list)):
            for j in range(i, len(unseen_tag_list)):
                keys = self.get_all_possible_tags(unseen_tag_list[i], unseen_tag_list[j])
                keys_set.update(keys)
        for key in keys_set:
            self.unseen_confusion_matrix.setdefault(key, 0)

        return \
            {
                'Miss per word': miss,
                'Hit per word': hit,
                'Accuracy per word': float(hit)/float(miss+hit),
                'confusion_matrix per word': self.confusion_matrix
             }

    def write_result_doc(self):

        file_name = self.write_file_name
        lines_count = len(self.predict_dict)
        with open(file_name, 'w') as f:
            for sentence_index, sequence_list in self.predict_dict.items():
                sentence_len = len(sequence_list)
                for word_index, word_tag_string in enumerate(sequence_list):
                    sep = ' '
                    if word_index+1 == sentence_len and sentence_index+1 < lines_count:  # if EOL but not EOF, add \n
                        sep = '\n'
                    f.write("{0}{1}".format(word_tag_string, sep))
        return

    def write_confusion_doc(self):
        """
            build confusion matrix doc
            build structure of line and columns
        """

        file_name = self.confusion_file_name
        top_k_tags_set, confusion_matrix_to_write = self.get_most_missed_tags()
        
        book = xlwt.Workbook(encoding="utf-8")
        # top-K confusion matrix
        self.create_confusion_sheet(book, top_k_tags_set, confusion_matrix_to_write, "Confusion Matrix")
        # unseen confusion matrix
        unseen_tags_list = sorted(self.unseen_tags_set)
        self.create_confusion_sheet(book, unseen_tags_list, self.unseen_confusion_matrix, "Unseen Confusion Matrix")
        book.save(file_name)

    def create_confusion_sheet(self, book, tag_list, confusion_matrix_to_write, sheet_name):
        """
        this method creates a new confusion matrix sheet by the name sheet_name
        :param sheet_name:
        :param book: the excel workbook object
        :param tag_list: list of all the tags
        :param confusion_matrix_to_write:
        :return: None
        """
        sheet1 = book.add_sheet(sheet_name)
        # regular pattern
        pattern = xlwt.Pattern()
        pattern.pattern = xlwt.Pattern.SOLID_PATTERN
        pattern.pattern_fore_colour = 22
        style = xlwt.XFStyle()
        style.pattern = pattern
        # mistakes pattern
        pattern_mistake = xlwt.Pattern()
        pattern_mistake.pattern = xlwt.Pattern.SOLID_PATTERN
        pattern_mistake.pattern_fore_colour = 2
        style_mistake = xlwt.XFStyle()
        style_mistake.pattern = pattern_mistake
        # correct pattern
        pattern_hit = xlwt.Pattern()
        pattern_hit.pattern = xlwt.Pattern.SOLID_PATTERN
        pattern_hit.pattern_fore_colour = 3
        style_hit = xlwt.XFStyle()
        style_hit.pattern = pattern_hit

        sheet1.write(0, 0, ' ', style)
        for idx_tag, cur_tag in enumerate(tag_list):
            sheet1.write(0, idx_tag + 1, cur_tag, style)
        for row_tag_idx, row_tag in enumerate(tag_list):
            sheet1.write(row_tag_idx + 1, 0, row_tag, style)
            for col_tag_idx, col_tag in enumerate(tag_list):
                cur_value = confusion_matrix_to_write["{0}_{1}".format(row_tag, col_tag)]
                if cur_value == 0:
                    sheet1.write(row_tag_idx + 1, col_tag_idx + 1, str(cur_value))
                else:
                    if row_tag_idx == col_tag_idx:
                        sheet1.write(row_tag_idx + 1, col_tag_idx + 1, str(cur_value), style_hit)
                    else:
                        sheet1.write(row_tag_idx + 1, col_tag_idx + 1, str(cur_value), style_mistake)
        return

    def get_most_missed_tags(self):
        top_tags_list = sorted(self.misses_matrix.items(), key=lambda x: x[1], reverse=True)[:self.k]
        tag_set = set()
        top_k_confusion_matrix = {}
        tags_keys = set()
        for key, val in top_tags_list:
            gold, predict = key.split('_')
            tag_set.update((gold, predict))
        tag_set = sorted(tag_set)
        # todo: check whether we can cut the loops
        for i in range(len(tag_set)):
            for j in range(i, len(tag_set)):
                keys = self.get_all_possible_tags(tag_set[i], tag_set[j])
                tags_keys.update(keys)
        for key in tags_keys:
            value = self.confusion_matrix.get(key, 0)
            top_k_confusion_matrix.update({key: value})
        return tag_set, top_k_confusion_matrix

    def get_all_possible_tags(self, gold, predict):
        """
        this method generates all possible combination of a given two tags gold and predict
        :param gold: first tag
        :param predict: second tag
        :return:  a set of tags
        """
        keys = eval("{{'{tag_1}_{tag_1}','{tag_1}_{tag_2}','{tag_2}_{tag_1}','{tag_2}_{tag_2}'}}"
                    .format(tag_1=gold, tag_2=predict))
        return keys

    def add_missing_tags(self, gold_tag, predict_tag):
        res = self.get_all_possible_tags(gold_tag, predict_tag)
        for confusion_matrix_key in res:
            self.confusion_matrix.setdefault(confusion_matrix_key, 0)
            self.misses_matrix.setdefault(confusion_matrix_key, 0)
        return

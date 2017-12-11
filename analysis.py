import csv
import pickle
import numpy as np
from datetime import  datetime
import os


class Analysis:
    """
    This class intend to analyse the results of the training,
    e.g. check which feature are not needed due to infrequency and lack of added value
    """

    def __init__(self):
        self.epsilon = 0

    def check_redundant_classifier_entries(self):
        """
        this method gets the weight vector, and checking which entries smaller than epsilon
        :return: redundant features
        """
        weight = pickle.load(open(os.path.join('resources', 'w_vec.pkl_11_12_2017_11_42_44'), 'rb'))
        features_vector_mapping = {}
        features_vector_mapping_file = csv.reader(open(
            os.path.join('dict',  'feature_100_feature_101_feature_102_feature_103_feature_104_feature_105_feature_106_feature_107_feature_108_feature_109_feature_110_feature_111_features_vector_mapping.csv'), 'r'))
        for row in features_vector_mapping_file:
            if len(row) != 0:
                key, val = row
                features_vector_mapping.update({int(key): val})
        candidate_for_drop = []
        for index, value in enumerate(weight.x):
            if value <= self.epsilon:
                candidate_for_drop.append(features_vector_mapping[index])
        print("Features to drop:")
        for feature in candidate_for_drop:
            print(feature)

        large_weights = [[features_vector_mapping[index], value] for index, value in enumerate(weight.x) if abs(value) > 1]
        analysis_file_name = os.path.join('analysis', 'large_weight_vec.csv')
        with open(analysis_file_name, "w") as summary_file:
            analyze = csv.writer(summary_file)
            for row in large_weights:
                analyze.writerow(row)
        return candidate_for_drop, large_weights


if __name__ == '__main__':
    analysis = Analysis()
    analysis.check_redundant_classifier_entries()

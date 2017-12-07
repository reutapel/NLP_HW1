import csv
import pickle
import numpy as np
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
        weight = pickle.load(open(os.path.join('resources', 'w_vec.pkl'), 'rb'))
        features_vector_mapping = {}
        features_vector_mapping_file = csv.reader(open(
            os.path.join('dict',  'feature_100_feature_103_feature_104_features_vector_mapping.csv'), 'r'))
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
        return candidate_for_drop


if __name__ == '__main__':
    analysis = Analysis()
    analysis.check_redundant_classifier_entries()

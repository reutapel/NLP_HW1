import pickle
import numpy as np


class Analysis:
    """
    This class intend to analyse the results of the training,
    e.g. check which feature are not needed due to infrequency and lack of added value
    """

    def __init__(self):
        self.epsilon = 0
        pass

    def check_redundant_classifier_entries(self):
        """
        this method gets the weight vector, and checking which entries smaller than epsilon
        :return: redundant features
        """
        weight = pickle.load(open('resources\\w_vec.pkl', 'rb'))
        features_vector_mapping = pickle.load(open('resources\\feature_vector_mapping.pkl', 'rb'))
        candidate_for_drop = []
        for index, value in enumerate(weight):
            if value <= self.epsilon:
                candidate_for_drop.append(features_vector_mapping[index])
        print("Features to drop:")
        for feature in candidate_for_drop:
            print(feature)
        return candidate_for_drop


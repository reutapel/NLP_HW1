import csv
import pickle
import numpy as np
from datetime import  datetime
import os
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt




class Analysis:
    """
    This class intend to analyse the results of the training,
    e.g. check which feature are not needed due to infrequency and lack of added value
    """

    def __init__(self):
        self.epsilon = 0

    def check_redundant_classifier_entries(self, file_name=None):
        """
        this method gets the weight vector, and checking which entries smaller than epsilon
        :return: redundant features
        """
        if not file_name:
            file_name = os.path.join('resources', 'w_vec.pkl_11_12_2017_11_42_44')
        weight = pickle.load(open(file_name, 'rb'))
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

    def create_graph(self,file_name):
        """

        :param file_name:
        :return:
        """
        data = pickle.load(open(file_name, 'rb'))
        gradient = [val[0] for key, val in data.items()]
        loss = [val[1] for key, val in data.items()]
        fig, ax1 = plt.subplots()

        grad = ax1.plot(range(len(data)), gradient, '-ob', label='gradient')
        ax1.set_ylabel('gradient', color='b')
        ax1.tick_params('y', colors='b')
        ax2 = ax1.twinx()
        loss = ax2.plot(range(len(data)), loss, '-or', label='loss')
        ax2.set_ylabel('loss', color='r')
        ax2.tick_params('y', colors='r')
        ax1.set_title('gradient loss graph')
        lines = grad + loss
        lbls = [l.get_label() for l in lines]
        ax1.legend(lines, lbls, loc=0)
        plt.show()


if __name__ == '__main__':
    analysis = Analysis()
    root = tk.Tk()
    root.withdraw()
    # file_path = filedialog.askopenfilename()
    # analysis.check_redundant_classifier_entries(file_path)
    file_path = filedialog.askopenfilename()
    analysis.create_graph(file_path)




import pickle
from sklearn import preprocessing
import numpy as np
import pandas as pd

class GetPredictions():
    def __init__(self):
        self.model = pickle.load(open("models/kmeans_model.pkl", "rb"))
        self.labelmaker = preprocessing.LabelBinarizer()

    def preprocess(self, data_dict):
        preprocessed_data = [] 
        list_of_answers = data_dict.values()
        for answer in list_of_answers:
            preprocessed_data.append(list(np.reshape(self.labelmaker.fit_transform(answer), (1,10)))[0])
        preprocessed_data = pd.DataFrame(preprocessed_data)
        return preprocessed_data
    
    def predict(self, data):
        processed_data = self.preprocess(data)
        self.model.fit(processed_data)
        cluster_assignments = self.model.labels_
        results = dict(zip(data.keys(), cluster_assignments))
        return results


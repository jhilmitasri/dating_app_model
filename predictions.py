import pickle
from sklearn import preprocessing
import numpy as np
import pandas as pd

class GetPredictions():
    def __init__(self):
        self.model = pickle.load(open("models/svc_classifier.pkl", "rb"))
        self.labelmaker = preprocessing.LabelBinarizer()

    def preprocess(self, data_dict):
        preprocessed_data = [] 
        list_of_answers = data_dict.values()
        for answer in list_of_answers:
            answer = answer + ["a", "b"]
            preprocessed_data.append(list(np.reshape(self.labelmaker.fit_transform(answer)[:10], (1,10)))[0])
        # preprocessed_data = pd.DataFrame(preprocessed_data)
        return preprocessed_data
    
    def predict(self, data):
        probabilities = {}
        processed_data = self.preprocess(data)
        for (person, person_data) in zip(data.keys(), processed_data):
            result = self.model.predict_proba([person_data])
            probabilities[person] = list((result)[0])
        return probabilities


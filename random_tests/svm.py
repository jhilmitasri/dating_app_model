import pandas as pd
import numpy as np
import pickle

initial_data = {"Q_no":[1,2,3,4,5,6,7,8,9,10], 
                "P1":["0", "0", "0", "0", "0", "0", "0", "0", "0", "0"], 
                "P2":["1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}
df = pd.DataFrame(data=initial_data)
df.drop('Q_no', axis=1, inplace=True)

from sklearn.svm import SVC
svclassifier = SVC(kernel='linear', probability=True)
svclassifier.fit([["0", "0", "0", "0", "0", "0", "0", "0", "0", "0"], ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]], ["1","0"])
pickle.dump(svclassifier, open("models/svc_classifier.pkl", "wb"))

y_pred = svclassifier.predict_proba([["1", "1", "1", "0", "1", "1", "1", "0", "0", "1"]])
print(y_pred)
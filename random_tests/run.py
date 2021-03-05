import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import KMeans
from sklearn import preprocessing

model_name = "kmeans"

model = pickle.load(open("kmeans_model.pkl", "rb"))
ans1 = ["a", "a", "a", "a", "a", "a", "a", "b", "b", "b"]
ans2 = ["b", "b", "b", "b", "b", "a", "b", "b", "a", "b"]
ans3 = ["a", "a", "b", "b", "a", "a", "b", "a", "b", "a"]
ans4 = ["b", "b", "b", "b", "b", "b", "b", "b", "a", "b"]
labelmaker = preprocessing.LabelBinarizer()
print(list(np.reshape(labelmaker.fit_transform(ans1), (1,10)))[0])
list_of_answers = [ 
                    list(np.reshape(labelmaker.fit_transform(ans1), (1,10)))[0], 
                    list(np.reshape(labelmaker.fit_transform(ans2), (1,10)))[0], 
                    list(np.reshape(labelmaker.fit_transform(ans3), (1,10)))[0], 
                    list(np.reshape(labelmaker.fit_transform(ans4), (1,10)))[0]
                    ] 
print(list_of_answers)
list_of_answers = pd.DataFrame(list_of_answers)
# list_of_answers = list_of_answers.transpose()
print(list_of_answers)

if(model_name == "kmeans"):
    model.fit(list_of_answers)
    cluster_assignments = model.labels_
    cluster_centers = model.cluster_centers_
    inertia = model.inertia_
    list_of_answers['Cluster #'] = cluster_assignments

    print("\nResult: \n", list_of_answers)
    print("\nCluster centers: \n", cluster_centers)
    print("\nCluster assignments: \n", cluster_assignments)

elif(model_name == "hac"):
    result = model.fit_predict(list_of_answers)
    print("\n Init result: ", result)   
    cluster_assignments = model.labels_
    list_of_answers['Cluster #'] = cluster_assignments
    print("Result: \n", list_of_answers)
    print("\n Distances: \n", model.values)


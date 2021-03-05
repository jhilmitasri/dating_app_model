import pandas as pd
from sklearn.cluster import KMeans
import pickle

model = pickle.load(open("models/kmeans_model.pkl", "rb"))
testdf = pd.DataFrame(columns=["T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9"])
testlists = []
for i in range(0,11):
    testlist = []
    for i in range(0,i):
        testlist.append(0)
    for i in range(0,10-len(testlist)):
        testlist.append(1)    
    testlists.append(testlist)

testdf = pd.DataFrame(testlists)

swapped_test_df = testdf.swapaxes("index", "columns")
print("\n Arranged Data for Test:\n")
print(swapped_test_df)


model.fit(swapped_test_df)
cluster_assignments = model.labels_
cluster_centers = model.cluster_centers_
swapped_test_df['Cluster #'] = cluster_assignments
print("\n\nResult: \n", swapped_test_df)

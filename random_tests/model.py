import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import preprocessing

labelmaker = preprocessing.LabelBinarizer()
dp = labelmaker.fit_transform(['yes', 'no', 'no', 'yes'])
initial_data = {"Q_no":[1,2,3,4,5,6,7,8,9,10], "P1":["0", "0", "0", "0", "0", "0", "0", "0", "0", "0"], "P2":["1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}
df = pd.DataFrame(data=initial_data)
df.drop('Q_no', axis=1, inplace=True)
swapped_df = df.swapaxes("index", "columns") 
print(swapped_df)
df = swapped_df
# vectorizer = CountVectorizer()
# x = vectorizer.fit_transform(df['Q_no'])
# df_wrds = pd.DataFrame(x.toarray(), columns=vectorizer.get_feature_names())
# new_df = pd.concat([df, df_wrds], axis=1)

print(df)

# Instantiating HAC
hac = AgglomerativeClustering(n_clusters=2)

# Fitting
hac.fit(df)
pickle.dump(hac, open("models/hac_model.pkl", "wb"))
# Getting cluster assignments
cluster_assignments = hac.labels_

## KMeans Clustering with different number of clusters
# k_means = KMeans(n_clusters=2)

# k_means.fit(df)
# pickle.dump(k_means, open("models/kmeans_model.pkl", "wb"))


# cluster_assignments = k_means.predict(df)

# Assigning the clusters to each profile
df['Cluster #'] = cluster_assignments

# Viewing the dating profiles with cluster assignments
print(df)
testdf = pd.DataFrame(columns=["T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9"])
testlists = []
for i in range(0,11):
    testlist = []
    for i in range(0,i):
        testlist.append(0)
    for i in range(0,10-len(testlist)):
        testlist.append(1)
    
    # testdf["{}".format(i)] == testlist
    testlists.append(testlist)

testno = 0
# for occurence in testlists:
#     print(testno, len(occurence))
#     testdf["T{}".format(testno)] == occurence
#     testno += 1

testdf = pd.DataFrame(testlists)

# swapped_test_df = testdf.swapaxes("index", "columns")
swapped_test_df = testdf.transpose()
print("Final test")
print(swapped_test_df)

# Fitting
result = hac.fit_predict(swapped_test_df)
print("\n Init result: ", result)
# k_means.fit(swapped_test_df)

# Getting cluster assignments
cluster_assignments = hac.labels_

# cluster_assignments = k_means.labels_
# cluster_centers = k_means.cluster_centers_

# Assigning the clusters to each profile
swapped_test_df['Cluster #'] = cluster_assignments

print("Result: \n", swapped_test_df)
print("\n Distances: \n", hac.distances_)
# print(cluster_centers)
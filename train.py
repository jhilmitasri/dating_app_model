import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import KMeans, AgglomerativeClustering

initial_data = {"Q_no":[1,2,3,4,5,6,7,8,9,10], 
                "P1":["0", "0", "0", "0", "0", "0", "0", "0", "0", "0"], 
                "P2":["1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}
df = pd.DataFrame(data=initial_data)
df.drop('Q_no', axis=1, inplace=True)
swapped_df = df.swapaxes("index", "columns") 
print("\n\tInitial Data for Training: \n")
print(swapped_df)
df = swapped_df
k_means = KMeans(n_clusters=2)
k_means.fit(df)
# pickle.dump(k_means, open("kmeans_model_ittr2.pkl", "wb"))
cluster_assignments = k_means.predict(df)

df['Cluster #'] = cluster_assignments
print("\n\tResult after Training: \n")
print(df)


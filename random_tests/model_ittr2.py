import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn import preprocessing

initial_data = np.array([[0,0],[1,1]])
kmeans = KMeans(n_clusters=2, random_state=0).fit(initial_data)
result = kmeans.predict([[0, 1], [1, 0], [0, 0], [1, 1]])
print("Result: ", result)
print("Cluster centers: ", kmeans.cluster_centers_)

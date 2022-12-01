#-------------------------------------------------------------------------
# AUTHOR: Siva Charan
# FILENAME: clustering.py
# SPECIFICATION: K-means clustering with silhouette scoring
# FOR: CS 5990- Assignment #5
# TIME SPENT: 1 hr
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import metrics

df = pd.read_csv('training_data.csv', sep=',', header=None) #reading the data by using Pandas library

#assign your training data to X_training feature matrix
X_training = np.array(df.values)
max_sil_score = 0
max_k = 0
plot_x_data = []
plot_y_data = []
for k in range(2, 21):
#run kmeans testing different k values from 2 until 20 clusters
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_training)

     #for each k, calculate the silhouette_coefficient by using: 
    sil_score = silhouette_score(X_training, kmeans.labels_)
    if sil_score > max_sil_score:
        max_sil_score = sil_score
        max_k = k
    plot_x_data.append(k)
    plot_y_data.append(sil_score)

#plot the value of the silhouette_coefficient for each k value of kmeans so that we can see the best k
plt.plot(plot_x_data, plot_y_data)

#reading the validation data (clusters) by using Pandas library
vdf = pd.read_csv('testing_data.csv', sep=',', header=None)

#assign your data labels to vector labels (you might need to reshape the row vector to a column vector)
labels = np.array(vdf.values).reshape(1,len(vdf.values))[0]


#Calculate and print the Homogeneity of this kmeans clustering
print("K-Means Homogeneity Score = " + metrics.homogeneity_score(labels, kmeans.labels_).__str__())
#--> add your Python code here

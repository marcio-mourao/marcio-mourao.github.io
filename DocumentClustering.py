#This is a simplified version of the document clustering discussed at 
#http://scikit-learn.org/stable/auto_examples/text/document_clustering.html
#sphx-glr-auto-examples-text-document-clustering-py

from sklearn.datasets import fetch_20newsgroups
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics

#Import documents corresponding to the given categories and convert the
#documents to bags of words using TfdfVectorizer
categories = ['alt.atheism','talk.religion.misc','comp.graphics','sci.space']
dataset = fetch_20newsgroups(subset='all', categories=categories,shuffle=True,
                             random_state=42)
labels=dataset.target
vectorizer = TfidfVectorizer(max_df = 0.5, max_features=1000, min_df=2,
                             stop_words='english',use_idf=True)
#X consists of the bags of words
X = vectorizer.fit_transform(dataset.data)

#Perform Kmeans clustering for differing numbers of clusters and calculate the
#silhouette statistic, V measure, and inertia
silhouettes = []
V = []
inertias = []
cluster_sizes = range(2,11) #Sizes ranging from 2 to 10
for cluster_size in cluster_sizes:
    km=KMeans(n_clusters = cluster_size, init='k-means++', max_iter=100,
              n_init=10)
    km.fit(X)
    silhouettes.append(metrics.silhouette_score(X,km.labels_))
    V.append(metrics.v_measure_score(labels,km.labels_))
    inertias.append(km.inertia_)
print(silhouettes)
print(V)
print(inertias)

#Plot the scores
import matplotlib.pyplot as plt
plt.scatter(cluster_sizes,V)
#Connects the points in the ROC plot with lines
plt.plot(cluster_sizes,V)
#Adds diagonal line
plt.scatter(cluster_sizes,silhouettes)
plt.plot(cluster_sizes,silhouettes)
#Sets up the axis labels
plt.xlabel('Cluster size')
plt.ylabel('Score')
plt.ylim(0,1.01)
plt.text(6,0.2,"Silhouette score",color="green")
plt.text(6,0.6,"V-measure",color="blue")
#Displays the plot
plt.show()

#Determine how the V-measure changes as the number of features changes
V = []
features = [10,20,40,80,160,320,640,1280,2560,5120]
for num in features:
    vectorizer = TfidfVectorizer(max_df = 0.5, max_features=num, min_df=2,
                             stop_words='english',use_idf=True)
    X = vectorizer.fit_transform(dataset.data)
    km=KMeans(n_clusters = 4, init='random', max_iter=200,
              n_init=20)
    km.fit(X)
    V.append(metrics.v_measure_score(labels,km.labels_))
    
#Plot the change in V-measure as the number of featuers changes
plt.scatter(features,V)
plt.plot(features,V)
plt.xlabel("Number of features")
plt.ylabel("V-measure")
plt.show()
    
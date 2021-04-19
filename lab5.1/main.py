import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans

col_names = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Class']
data = pd.read_csv('iris.csv', delimiter=',', header = None, names = col_names )
print(data.head())

le = preprocessing.LabelEncoder()

feature_cols = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width']
data = data.apply(le.fit_transform)
X = data[feature_cols]
y_true = data.Class
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], s = 20);
plt.show()

kmeans = KMeans(n_clusters = 3)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c = y_kmeans, s = 20, cmap = 'summer')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c = 'blue', s = 100, alpha = 0.9);
plt.show()

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sb 
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler 
from sklearn.cluster import KMeans # K-means algorithm

col_names = ['Row', 'CustomerId', 'Age', 'Edu', 'YearsEmployed', 'Income', 'CardDebt', 'OtherDebt', 'Defaulted', 'DebtIncomeRatio']
df = pd.read_csv('customers.csv', delimiter=',', header = None, names = col_names )
df.head()

X = df[col_names]
X = np.nan_to_num(X)

sc = StandardScaler()
cluster_data = sc.fit_transform(X)

print('Cluster data samples : ', cluster_data[:2])

clusters = 3
model = KMeans(n_clusters = clusters)
model.fit(X)
y = model.predict(X)

fig = plt.figure(1)
plt.clf()
ax = Axes3D(fig, 
            rect = [0, 0, .95, 1], 
            elev = 48, 
            azim = 134)

plt.cla()
ax.scatter(df['Edu'], df['Age'], df['Income'], 
           c = y, 
           s = 200, 
           cmap = 'spring', 
           alpha = 0.5, 
           edgecolor = 'darkgrey')

ax.set_xlabel('Education', 
              fontsize = 16)
ax.set_ylabel('Age', 
              fontsize = 16)
ax.set_zlabel('Income', 
              fontsize = 16)

plt.savefig('3d_plot.png')
plt.show()

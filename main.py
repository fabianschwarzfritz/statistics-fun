# Dependencies

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

#%matplotlib inline

# Implementation

# Read the csv file. File is read in as a 'dataframe' data-type.
df = pd.read_csv('simple.csv')

# Print the dataframe
print('### HEAD')
print(df)

# Apply the k-means clustering.
print('')
print('### K MEANS')
kmeans = KMeans(n_clusters=2).fit(df)
print('# Cluster labels')
print(kmeans.labels_)
print('# Cluster centers')
print(kmeans.cluster_centers_)

# Add the prediction column to the dataframe.
print('')
print('### Data with prediction')
df['prediction'] = kmeans.labels_
#print(df.head())
print(df)

# Visualize the datapoints of the csv
print('')
print('### Visualization')
plt.figure(figsize=(6,6))
x = df.iloc[:,0]
print(x)
y = df.iloc[:,1]
print(y)
plt.scatter(x, y)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

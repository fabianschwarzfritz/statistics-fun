# Dependencies

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt

#%matplotlib inline

# Implementation

# Read the csv file. File is read in as a 'dataframe' data-type.
df_train = pd.read_csv('http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv')
df_test = pd.read_csv('http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv')

# Print the dataframe
print('### HEAD - TRAIN')
print(df_train.head())
print('### HEAD - TEST')
print(df_test.head())

# Clear the dataset
# The dataset has multiple null values for properties: Age, Cabin, Embarked
print('### Dataset clearance')
print(df_train.isna().sum())

# We can see that the dataset has null values for the Age and Emabrarked property.
# We can use the mean() value of the dataset to fill in the values.
df_train.fillna(df_train.mean(), inplace=True)
df_test.fillna(df_test.mean(), inplace=True)

# TODO how to clear the Cabin property
# The Cabin attribute is still missing. 

# We now try to determine the suvival count (cluster survival=1 and survival=0)
# based on the attributes: Pclass, Sex, SibSp, Parch
print(df_train[['Pclass', 'Survived']].groupby(['Pclass'])).mean()
print(df_train[['Sex', 'Survived']].groupby(['Sex'])).mean()
print(df_train[['SibSp', 'Survived']].groupby(['SibSp'])).mean()

# We can plot the survival by 'Age'
g = sns.FacetGrid(df_train, col='Survived')
g.map(plt.hist, 'Age', bins=20)
g.add_legend()
#plt.show()

# And we can plot the survival by 'Pclass'
g2 = sns.FacetGrid(df_train, col='Survived', row='Pclass')
g2.map(plt.hist, 'Age', bins=20)
g2.add_legend()
#plt.show()

# Kmeans 
print('### K-Means')
#df_train = df_train.head(30)

# Save the survives property for later comparistion
y = np.array(df_train['Survived']) # the survived property

# The only non-numeric feature is the 'Sex' attribute. We use labels
# to replace the possible values with numeric values.
le = LabelEncoder()
le.fit(df_train['Sex'])
df_train['Sex'] = le.transform(df_train['Sex'])

# The dataset we want to use for kmeans
df_train = df_train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]

# Scale the values in the dataframe
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(df_train)

kmeans = KMeans(n_clusters=2).fit(x_scaled)
# Add the prediction as a new column
df_train['survival_prediction'] = kmeans.labels_
print("#######################################")
print(kmeans.labels_)
print(df_train)
print("#######################################")

# Calculate how much entries in one of the clusters
# correctly represent the survival count of the 
# passengers
correct = 0
for i in df_train.index:
    expected = df_train['survival_prediction'][i]
    actual = y[i]
    if expected == actual:
        correct += 1

rowcount = df_train['Pclass'].count()
print("Number of 'rows' in the dataset:     ", rowcount)
print("Number of 'correct' classified rows: ", correct)
print("Percentage of correct classified:    ", (correct*1.0)/(rowcount*1.0))

# With this block we can try to find the optimal number of k groups to cluster
# the data. In case we would not know that 2 is useful (survived and not-survived)
# we can see when the data changes less for increasing k. This is probably a good 
# value for k to cluster them.
# Of course, a closer look to the data or a decision of k because of human
# domain knowledge makes more sense.
for k in range(2, 20):
	print("# {}-means for dataset".format(k))
	# Apply k-means (apply to the scaled dataset)
	kmeans = KMeans(n_clusters=k)
	prediction = kmeans.fit_predict(x_scaled)

	score = silhouette_score(x_scaled, prediction, metric='euclidean')
	print("For n_clusters = {}, silhouette score is {})".format(k, score))

	# Print the inertia
	print(kmeans.inertia_)
	print("k: %s, cost: %s", k, kmeans.inertia_)


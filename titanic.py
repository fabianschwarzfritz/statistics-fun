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
print(df_train.head())
# The features like 'Name', 'Ticket', 'Cabin', 'Embarked' do not have any
# inpact on the dataset so we should drop them.
df_train = df_train.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)
# The only non-numeric feature is the 'Sex' attribute. We use labels
# to replace the possible values with numeric values.
le = LabelEncoder()
le.fit(df_train['Sex'])
le.fit(df_test['Sex'])
df_train['Sex'] = le.transform(df_train['Sex'])
df_test['Sex'] = le.transform(df_test['Sex'])
df_train.info()
print(df_train.head())

# Apply the k-means clustering.
print('### K MEANS')

# We have to drop the survived columnd from the train dataset.
x = np.array(df_train.drop(['Survived'], 1).astype(float)) # The test data set without 'survived' property
y = np.array(df_train['Survived']) # the survived property

kmeans = KMeans(n_clusters=2).fit(x)
print(kmeans.labels_)
print(kmeans.cluster_centers_)
df_train['survival_prediction'] = kmeans.labels_
df_train.info()
print(df_train)

# TODO open question: All the first rows are 1, all the second rows are 0 in regards to survival....


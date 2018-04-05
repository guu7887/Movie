#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 16:24:42 2018

@author: Bing Miu
"""
# Movie Recommendation Lab
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import seaborn as sns
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold # import KFold
from sklearn import feature_selection
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.linear_model import LassoCV

#upload data
data_users = pd.read_csv("users.csv")
data_movies = pd.read_csv("movies.tsv", sep='\t')
data_ratings = pd.read_csv("ratings.csv")
data_all = pd.read_csv("allData.tsv", sep='\t')

#view data
data_all.head()

# Correlation matrix
d = data_all

# Compute the correlation matrix
corr = d.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


#view data
data_all.head()
data_all.columns
data_all.info()

#analyze data

data_all['age'].describe()
data_all['gender'].describe()
#histogram
sns.distplot(data_all['rating'])
#data_all.plot.scatter(x='age', y='rating')
fig = sns.boxplot(x='age', y='rating',data=data_all)
#correlation matrix
corrmat = data_all.corr()
sns.heatmap(corrmat, square=True);

##scatterplot
#sns.set()
#sns.pairplot(data_all, size = 2.5)
#plt.show();


#missing data
data_all.count()
data_all.isnull().sum()

# Create a set of dummy variables from the gender and genre variable
df_gender = pd.get_dummies(data_all['gender'])
df_genre1 = pd.get_dummies(data_all['genre1'])
df_genre2 = pd.get_dummies(data_all['genre2'])
df_genre3 = pd.get_dummies(data_all['genre3'])

df_genre = pd.concat([df_genre1,df_genre2,df_genre3]).groupby(level=0).any().astype(int)

data_new = pd.concat([data_all[['userID', 'age', 'movieID', 'name', 'year']], df_gender, df_genre,data_all['rating']], axis=1)

data_X = pd.concat([data_all[['age', 'year']], df_gender, df_genre], axis=1) # or w/o 'year'???
data_y = data_all['rating']

# create training and testing vars
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.2, random_state=1234)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)

# create Cross-Validation K-Fold
kf = KFold(n_splits=10, random_state=1234, shuffle=True)
for train_index, test_index in kf.split(X_train):
    X_CV_train, X_CV_test = X_train.iloc[train_index], X_train.iloc[test_index]
    y_CV_train, y_CV_test = y_train.iloc[train_index], y_train.iloc[test_index]
    # implement regression part

# Create linear regression object
lm = linear_model.LinearRegression()

# Train the model using the training sets
lm.fit(X_train, y_train)



lasso=Lasso(alpha=0.01, fit_intercept=False)
lasso.fit(X_train, y_train)
y_est=lasso.predict(X_test)
mse=np.mean(np.square(y_test-y_est))
print(mse)
print(lasso.coef_)

ridge=Ridge(alpha=0.1, fit_intercept=False)
ridge.fit(X_train, y_train)
y_est=ridge.predict(X_test)
mse=np.mean(np.square(y_test-y_est))
print(mse)
print(ridge.coef_)







### test#######

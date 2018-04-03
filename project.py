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

# Create a set of dummy variables from the genre variable
df_genre1 = pd.get_dummies(data_all['genre1'])
df_genre2 = pd.get_dummies(data_all['genre2'])
df_genre3 = pd.get_dummies(data_all['genre3'])

df_genre = pd.concat([df_genre1,df_genre2,df_genre3]).groupby(level=0).any().astype(int)

data_new = pd.concat([data_all[['userID', 'age', 'gender', 'movieID', 'name', 'year']], df_genre,data_all['rating']], axis=1)





### test#######

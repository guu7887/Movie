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
import statsmodels.formula.api as smf
import statsmodels.api as sm

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

#Drop M (only include F in the model)
X_train = X_train.drop ('M', 1)
X_test = X_test.drop('M', 1)    

#Linear regression
# Train the model using the training sets
X = sm.add_constant(X_train)
olsmod = sm.OLS(y_train, X).fit()            
print(olsmod.summary())

# Test the model using the test sets
X2 = sm.add_constant(X_test)
ypred = olsmod.predict(X2)

# Test linear regression model using MSE 
np.mean((ypred - y_test)**2)

#Stepwise Selection:

#Drop Musical

Xa = X.drop('Musical', 1)
X2a= X2.drop('Musical', 1)
olsmod = sm.OLS(y_train, Xa).fit()
print(olsmod.summary())
ypred = olsmod.predict(X2a)
MSE = np.mean((ypred - y_test)**2)
print(MSE)

#Drop Musical+ Mystery
Xb = Xa.drop('Mystery', 1)
X2b= X2a.drop('Mystery', 1)
olsmod = sm.OLS(y_train, Xb).fit()
print(olsmod.summary())
ypred = olsmod.predict(X2b)
MSE = np.mean((ypred - y_test)**2)
print(MSE)

#Drop Musical+ Mystery+Adventure
Xc = Xb.drop('Adventure', 1)
X2c= X2b.drop('Adventure', 1)
olsmod = sm.OLS(y_train, Xc).fit()
print(olsmod.summary())
ypred = olsmod.predict(X2c)
MSE = np.mean((ypred - y_test)**2)
print(MSE)

#Drop Musical+ Mystery+Adventure+F
Xd = Xc.drop('F', 1)
X2d= X2c.drop('F', 1)
olsmod = sm.OLS(y_train, Xd).fit()
print(olsmod.summary())
ypred = olsmod.predict(X2d)
MSE = np.mean((ypred - y_test)**2)
print(MSE)

#Drop Musical+ Mystery+Adventure+F+age

Xe = Xd.drop('age', 1)
X2e= X2d.drop('age', 1)
olsmod = sm.OLS(y_train, Xe).fit()
print(olsmod.summary())
ypred = olsmod.predict(X2e)
MSE = np.mean((ypred - y_test)**2)
print(MSE)


alphas = np.logspace(-5, -2, 1000)
lassocv = linear_model.LassoCV(alphas=alphas, cv=10, random_state=1234)
lassocv.fit(data_X, data_y)
lassocv_score = lassocv.score(data_X, data_y)
lassocv_alpha = lassocv.alpha_
lassocv_alpha


alphas = np.linspace(9,11,150)
ridgecv = linear_model.RidgeCV(alphas=alphas, scoring=None, cv=10)
ridgecv.fit(data_X, data_y)
ridgecv_score = ridgecv.score(data_X, data_y)
ridgecv_alpha = ridgecv.alpha_
ridgecv_alpha

lasso=Lasso(alpha=0.0003)
lasso.fit(X_train, y_train)
y_est=lasso.predict(X_test)
mse=np.mean(np.square(y_test-y_est))
print(mse)
print(lasso.coef_)

ridge=Ridge(alpha=9.7)
ridge.fit(X_train, y_train)
y_est=ridge.predict(X_test)
mse=np.mean(np.square(y_test-y_est))
print(mse)
print(ridge.coef_)


rating_year=data_all.groupby('year')
means=rating_year.mean()
plt.scatter(means.index, means['rating'])
plt.show()

rating_age=data_all.groupby('age')
means=rating_age.mean()
plt.scatter(means.index, means['rating'])
plt.show()

# Matrix Factorization
df = data_all.pivot(index='userID', columns='movieID', values='rating')

"""
@INPUT:
    R     : a matrix to be factorized, dimension N x M
    P     : an initial matrix of dimension N x K
    Q     : an initial matrix of dimension M x K
    K     : the number of latent features
    steps : the maximum number of steps to perform the optimisation
    alpha : the learning rate
    beta  : the regularization parameter
@OUTPUT:
    the final matrices P and Q
"""

def matrix_factorization(data, df, R, P, Q, K, steps=100, alpha=0.01, beta=0.25):
    indices = range(data.shape[0])
    indices_train, indices_test = train_test_split(indices, test_size=0.1, random_state=123)
    Q = Q.T
    e = 100*len(indices_test)
    P_temp = P.copy()
    Q_temp = Q.copy()
    for step in range(steps):
        for k in indices_train:
            uid = data.iloc[k,0]
            mid = data.iloc[k,3]
            i = df.index.get_loc(uid)
            j = df.columns.get_loc(mid)
            eij = R[i][j] - np.dot(P_temp[i,:],Q_temp[:,j])
            for k in range(K):
                P_temp[i][k] = max(P_temp[i][k] + alpha * (2 * eij * Q_temp[k][j] - beta * P_temp[i][k]),0)
                Q_temp[k][j] = max(Q_temp[k][j] + alpha * (2 * eij * P_temp[i][k] - beta * Q_temp[k][j]),0)

    
        #eR = np.dot(P,Q)
        print(step)

        e_temp = 0
        for k in indices_test:
            uid = data.iloc[k,0]
            mid = data.iloc[k,3]
            i = df.index.get_loc(uid)
            j = df.columns.get_loc(mid)
            e_temp = e_temp + pow(R[i][j] - np.dot(P_temp[i,:],Q_temp[:,j]), 2)
#                    for k in range(K):
#                        e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
        print(e_temp/len(indices_test))
#        if (e_temp/31620) < 0.1:
#            break
        if e_temp < e:
            if e-e_temp < 0.00001:
                break
            alpha = alpha*1.05
            P = P_temp.copy()
            Q = Q_temp.copy()
            e = e_temp
            temp = 0
        else:
            alpha = alpha*0.5
            P_temp = P.copy()
            Q_temp = Q.copy()
            temp += 1
        if temp == 3:
            break
    return P, Q.T

R = df.values

N = len(R)
M = len(R[0])
K = 15

np.random.seed(123)
P = np.random.rand(N,K)
Q = np.random.rand(M,K)

nP, nQ = matrix_factorization(data_all.iloc[indices_train], df, R, P, Q, K)
nR = np.dot(nP, nQ.T)

e = 0
r_test = []
r_pred = []
for k in indices_test:
    uid = data_all.iloc[k,0]
    mid = data_all.iloc[k,3]
    i = df.index.get_loc(uid)
    j = df.columns.get_loc(mid)
    e += (R[i,j]-nR[i,j])**2
    r_test.append(R[i,j]) 
    r_pred.append(nR[i,j])
    

mse = e/len(indices_test)

d = {'test': r_test, 'predict': r_pred}
df_res = pd.DataFrame(data=d)
df_res.groupby('test').mean()



### test#######

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 16:24:42 2018

@author: jcfeng
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
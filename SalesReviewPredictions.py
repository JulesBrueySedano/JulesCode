#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 20:35:48 2021

@author: jbs
"""

from pathlib import Path

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import matplotlib.pylab as plt
from dmba import plotDecisionTree, classificationSummary, regressionSummary


import heapq

from collections import defaultdict

import surprise

import pandas as pd

import matplotlib.pylab as plt

from mlxtend.frequent_patterns import apriori

from mlxtend.frequent_patterns import association_rules

 

from surprise import Dataset, Reader, KNNBasic

from surprise.model_selection import train_test_split

 

data_df = pd.read_csv("/Users/jbs/Downloads/Reviews.csv")




data_df = data_df.dropna()
data_df = data_df.drop(columns = ['ProfileName',  'HelpfulnessNumerator', 'HelpfulnessDenominator',   'Time', 'Summary',  'Text'])




#NEw Code

def get_top_n(predictions, n=10):
     byUser = defaultdict(list)
     for p in predictions:
        byUser[p.uid].append(p)
    
    # For each user, reduce predictions to top-n
     for uid, userPredictions in byUser.items():
        byUser[uid] = heapq.nlargest(n, userPredictions, key=lambda p: p.est)
     return byUser


reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(data_df[['Id', 'UserId', 'Score']], reader)

# Split into training and test set
trainset, testset = train_test_split(data, test_size=.1, random_state=1)

## User-based filtering
# compute cosine similarity between users 
sim_options = {'name': 'cosine', 'user_based': True}
algo = KNNBasic(sim_options=sim_options)
algo.fit(trainset)

# Than predict ratings for all pairs (u, i) that are NOT in the training set.
predictions = algo.test(testset)

top_n = get_top_n(predictions, n=4)

# Print the recommended items for each user
print()
print('Top-3 recommended items for each user')
for uid, user_ratings in list(top_n.items())[:5]:
    print('User {}'.format(uid))
    for prediction in user_ratings:
        print('  Item {0.iid} ({0.est:.2f})'.format(prediction), end='')
    print()
print()

    
## Item-based filtering
# compute cosine similarity between users 
sim_options = {'name': 'cosine', 'user_based': False}
algo = KNNBasic(sim_options=sim_options)
algo.fit(trainset)

# Than predict ratings for all pairs (u, i) that are NOT in the training set.
predictions = algo.test(testset)
top_n = get_top_n(predictions, n=4)

# Print the recommended items for each user
print()
print('Top-3 recommended items for each user')
for uid, user_ratings in list(top_n.items())[:5]:
    print('User {}'.format(uid))
    for prediction in user_ratings:
        print('  Item {0.iid} ({0.est:.2f})'.format(prediction), end='')
    print()
    
    
trainset = data.build_full_trainset()
sim_options = {'name': 'cosine', 'user_based': False}
algo = KNNBasic(sim_options=sim_options)
algo.fit(trainset)

# Predict rating for user 383 and item 7
algo.predict(383, 7)
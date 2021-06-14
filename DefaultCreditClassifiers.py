#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 10:48:48 2021

@author: jbs
"""



import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import matplotlib.pylab as plt
from dmba import plotDecisionTree, classificationSummary, regressionSummary



default_df = pd.read_csv("/Users/jbs/Downloads/default_credit.csv")




X = default_df.drop(columns=['Y'])

y = default_df['Y']




#further splot
train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.1, random_state=1)



#smaller tree 
smallClassTree = DecisionTreeClassifier(max_depth=30, min_samples_split=20, min_impurity_decrease=0.0001)
smallClassTree.fit(train_X, train_y)

#test the accuracy

classificationSummary(train_y, smallClassTree.predict(train_X))
classificationSummary(valid_y, smallClassTree.predict(valid_X))

plotDecisionTree(smallClassTree, feature_names=train_X.columns)






param_grid = {
    'max_depth': [10, 20, 30, 40], 
    'min_samples_split': [20, 40, 60, 80, 100], 
    'min_impurity_decrease': [0, 0.0005, 0.001, 0.005, 0.01], 
}
gridSearch = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5, n_jobs=-1)
gridSearch.fit(train_X, train_y)
print('Initial score: ', gridSearch.best_score_)
print('Initial parameters: ', gridSearch.best_params_)

# Adapt grid based on result from initial grid search
param_grid = {
    'max_depth': list(range(2, 16)), 
    'min_samples_split': list(range(10, 22)), 
    'min_impurity_decrease': [0.0009, 0.001, 0.0011], 
}
gridSearch = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5, n_jobs=-1)
gridSearch.fit(train_X, train_y)
print('Improved score: ', gridSearch.best_score_)
print('Improved parameters: ', gridSearch.best_params_)

bestClassTree = gridSearch.best_estimator_



#given param information

# Initial score:  0.8191851851851851
# Initial parameters:  {'max_depth': 10, 'min_impurity_decrease': 0.0005, 'min_samples_split': 20}
# Improved score:  0.8191851851851851
# Improved parameters:  {'max_depth': 2, 'min_impurity_decrease': 0.0011, 'min_samples_split': 10}

NewTree = DecisionTreeClassifier(max_depth=2, min_samples_split=10, min_impurity_decrease=0.0011)
NewTree.fit(train_X, train_y)

#test the accuracy

classificationSummary(train_y, NewTree.predict(train_X))
classificationSummary(valid_y, NewTree.predict(valid_X))

plotDecisionTree(NewTree, feature_names=train_X.columns)

#Results

# Confusion Matrix (Accuracy 0.8192)

#        Prediction
# Actual     0     1
#      0 20177   863
#      1  4019  1941
# Confusion Matrix (Accuracy 0.8233)

#        Prediction
# Actual    0    1
#      0 2234   90
#      1  440  236

## TRy again


NewTree1 = DecisionTreeClassifier(max_depth=3, min_samples_split=15, min_impurity_decrease=0.00045)
NewTree1.fit(train_X, train_y)

#test the accuracy

classificationSummary(train_y, NewTree1.predict(train_X))
classificationSummary(valid_y, NewTree1.predict(valid_X))

plotDecisionTree(NewTree1, feature_names=train_X.columns)



# Confusion Matrix (Accuracy 0.8216)

#        Prediction
# Actual     0     1
#      0 20018  1022
#      1  3796  2164
# Confusion Matrix (Accuracy 0.8233)

#        Prediction
# Actual    0    1
#      0 2209  115
#      1  415  261



NewTree2 = DecisionTreeClassifier(max_depth=2, min_samples_split=8, min_impurity_decrease=0.0033)
NewTree2.fit(train_X, train_y)

#test the accuracy

classificationSummary(train_y, NewTree2.predict(train_X))
classificationSummary(valid_y, NewTree2.predict(valid_X))

plotDecisionTree(NewTree2, feature_names=train_X.columns)




# Confusion Matrix (Accuracy 0.8192)

#        Prediction
# Actual     0     1
#      0 20177   863
#      1  4019  1941
# Confusion Matrix (Accuracy 0.8233)

#        Prediction
# Actual    0    1
#      0 2234   90
#      1  440  236

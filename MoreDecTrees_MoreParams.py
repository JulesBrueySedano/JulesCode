#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 12:04:02 2021

@author: jbs
"""


import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import matplotlib.pylab as plt
from dmba import plotDecisionTree, classificationSummary, regressionSummary



spam_df = pd.read_csv("/Users/jbs/Downloads/Spambase.csv")




X = spam_df.drop(columns=['Spam'])

y = spam_df['Spam']





#further splot
train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.3, random_state=1)


fullClassTree = DecisionTreeClassifier()
fullClassTree.fit(train_X, train_y)

#print the tree
 
#test the accuracy
classificationSummary(train_y, fullClassTree.predict(train_X))
classificationSummary(valid_y, fullClassTree.predict(valid_X))


print("Classes: {}".format(', '.join(classTree.classes_)))
plotDecisionTree(classTree, feature_names=spam_df.columns[:2], class_names=classTree.classes_)












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


# Initial score:  0.9127329192546583
# Initial parameters:  {'max_depth': 10, 'min_impurity_decrease': 0.001, 'min_samples_split': 20}


# Initial score:  0.9118012422360249
# Initial parameters:  {'max_depth': 10, 'min_impurity_decrease': 0.001, 'min_samples_split': 20}
# Improved score:  0.9173913043478261
# Improved parameters:  {'max_depth': 11, 'min_impurity_decrease': 0.0009, 'min_samples_split': 10}


NewTree = DecisionTreeClassifier(max_depth=11, min_samples_split=10, min_impurity_decrease=0.0009)
NewTree.fit(train_X, train_y)

#test the accuracy

classificationSummary(train_y, NewTree.predict(train_X))
classificationSummary(valid_y, NewTree.predict(valid_X))

plotDecisionTree(NewTree, feature_names=train_X.columns)



#Confusion Matrix (Accuracy 0.9472)
#        Prediction
# Actual    0    1
#      0 1902   45
#      1  125 1148
# Confusion Matrix (Accuracy 0.9146)

#        Prediction
# Actual   0   1
#      0 806  35
#      1  83 457


NewTree1 = DecisionTreeClassifier(max_depth=12, min_samples_split=9, min_impurity_decrease=0.00045)
NewTree1.fit(train_X, train_y)

#test the accuracy

classificationSummary(train_y, NewTree1.predict(train_X))
classificationSummary(valid_y, NewTree1.predict(valid_X))

plotDecisionTree(NewTree1, feature_names=train_X.columns)




# Confusion Matrix (Accuracy 0.9680)

#        Prediction
# Actual    0    1
#      0 1915   32
#      1   71 1202
# Confusion Matrix (Accuracy 0.9102)

#        Prediction
# Actual   0   1
#      0 787  54
#      1  70 470





#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 10:42:25 2021

@author: jbs
"""


from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import matplotlib.pylab as plt
from dmba import plotDecisionTree, classificationSummary, regressionSummary

bank_df = pd.read_csv("/Users/jbs/Downloads/UniversalBank.csv")

bank_df.drop(columns=['ID', 'ZIP Code'], inplace=True)


# split into training and validation
X = bank_df.drop(columns=['Personal Loan'])
y = bank_df['Personal Loan']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.40, random_state=3)


#Single Tree

defaultTree = DecisionTreeClassifier(random_state=1)
defaultTree.fit(X_train, y_train)

classificationSummary(y_train, defaultTree.predict(X_train))
classificationSummary(y_valid, defaultTree.predict(X_valid))



#KNN

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

grid_params ={'n_neighbors':[3,5,7,9],
              'weights':['uniform','distance']}

gs = GridSearchCV(KNeighborsClassifier(),grid_params,cv=5,n_jobs=-1,scoring='accuracy')
gs.fit(X_train, y_train)
print(gs.best_params_)
bestKnn= gs.best_estimator_

classificationSummary(y_train, bestKnn.predict(X_train))
classificationSummary(y_valid, bestKnn.predict(X_valid))



#Random Forest 

rf = RandomForestClassifier(n_estimators=100,max_depth=6)
rf.fit(X_train, y_train)

classificationSummary(y_train, rf.predict(X_train))
classificationSummary(y_valid, rf.predict(X_valid))


#Bagging

bagging = BaggingClassifier(DecisionTreeClassifier(random_state=1), 
                            n_estimators=100, random_state=1)
bagging.fit(X_train, y_train)

classificationSummary(y_train, bagging.predict(X_train))
classificationSummary(y_valid, bagging.predict(X_valid))


# Boosting 

boost = AdaBoostClassifier(DecisionTreeClassifier(random_state=1), n_estimators=100, random_state=1)
boost.fit(X_train, y_train)

classificationSummary(y_train, boost.predict(X_train))
classificationSummary(y_valid, boost.predict(X_valid))




#XGBoost


from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train, y_train)

classificationSummary(y_train, model.predict(X_train))
classificationSummary(y_valid, model.predict(X_valid))

#more

model = XGBClassifier(gamma=0.1)
model.fit(X_train, y_train)

classificationSummary(y_train, model.predict(X_train))
classificationSummary(y_valid, model.predict(X_valid))



#Gradient Boosting


boost = GradientBoostingClassifier()
boost.fit(X_train, y_train)

classificationSummary(y_train, boost.predict(X_train))
classificationSummary(y_valid, boost.predict(X_valid))















































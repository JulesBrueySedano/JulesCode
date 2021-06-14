#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 09:23:53 2021

@author: jbs
"""

pip install graphviz
pip install pydotplus
pip install sklearn
pip install pydot
pip install pandas


pip install dmba

from pathlib import Path

conda install pydotplus


conda install graphviz

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import matplotlib.pylab as plt
from dmba import plotDecisionTree, classificationSummary, regressionSummary


import pydotplus
import pandas as pd
from sklearn import tree
from io import StringIO
import pydot

mower_df = pd.read_csv("/Users/jbs/Downloads/RidingMowers.csv")



classTree = DecisionTreeClassifier(random_state=0, max_depth=1)

classTree.fit(mower_df.drop(columns=['Ownership']), mower_df['Ownership'])



#viz of the tree
print("Classes: {}".format(', '.join(classTree.classes_)))
plotDecisionTree(classTree, feature_names=mower_df.columns[:2], class_names=classTree.classes_)



classTree2 = DecisionTreeClassifier(random_state=0)
classTree2.fit(mower_df.drop(columns=['Ownership']), mower_df['Ownership'])

#another viz
print("Classes: {}".format(', '.join(classTree2.classes_)))
plotDecisionTree(classTree2, feature_names=mower_df.columns[:2], class_names=classTree2.classes_)


print(mower_df.head())

# new df test rain split example
bank_df = pd.read_csv("/Users/jbs/Downloads/UniversalBank.csv")



bank_df = bank_df.drop(columns=['ID', 'ZIP Code'])

#spliting

X = bank_df.drop(columns=['Personal Loan'])

y = bank_df['Personal Loan']

#further splot
train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.4, random_state=1)


fullClassTree = DecisionTreeClassifier()
fullClassTree.fit(train_X, train_y)

#print the tree
plotDecisionTree(fullClassTree, feature_names=train_X.columns)

#test the accuracy
classificationSummary(train_y, fullClassTree.predict(train_X))
classificationSummary(valid_y, fullClassTree.predict(valid_X))



#smaller tree 
smallClassTree = DecisionTreeClassifier(max_depth=30, min_samples_split=20, min_impurity_decrease=0.01)
smallClassTree.fit(train_X, train_y)

#test the accuracy

classificationSummary(train_y, smallClassTree.predict(train_X))
classificationSummary(valid_y, smallClassTree.predict(valid_X))




#new df


customer_df = pd.read_csv("/Users/jbs/Downloads/Customers_Changed.csv")

customer_df = customer_df.drop(columns=['ID'])


X = customer_df.drop(columns=['CHURN'])

y = customer_df['CHURN']

#further splot
train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.4, random_state=1)

custClassTree = DecisionTreeClassifier(max_depth=30, min_samples_split=20, min_impurity_decrease=0.01)
custClassTree.fit(train_X, train_y)

plotDecisionTree(custClassTree, feature_names=train_X.columns)


classificationSummary(train_y, custClassTree.predict(train_X))
classificationSummary(valid_y, custClassTree.predict(valid_X))

#Confusion Matrix (Accuracy 0.8349)
 #      Prediction
#Actual   0   1
#     0 402  95
#     1 110 635
#Confusion Matrix (Accuracy 0.8164)

#       Prediction
#Actual   0   1
#     0 252  55
#     1  97 424

#another attempt
custClassTree2 = DecisionTreeClassifier(max_depth=25, min_samples_split=15, min_impurity_decrease=0.01)
custClassTree2.fit(train_X, train_y)

#accuracy
classificationSummary(train_y, custClassTree2.predict(train_X))
classificationSummary(valid_y, custClassTree2.predict(valid_X))

# Confusion Matrix (Accuracy 0.8349)

#        Prediction
# Actual   0   1
#      0 402  95
#      1 110 635
# Confusion Matrix (Accuracy 0.8164)

#        Prediction
# Actual   0   1
#      0 252  55
#      1  97 424






#another attempt
custClassTree3 = DecisionTreeClassifier(max_depth=35, min_samples_split=24, min_impurity_decrease=0.01)
custClassTree3.fit(train_X, train_y)

#accuracy
classificationSummary(train_y, custClassTree3.predict(train_X))
classificationSummary(valid_y, custClassTree3.predict(valid_X))


#same as above



#another attempt
custClassTree4 = DecisionTreeClassifier(max_depth=15, min_samples_split=10, min_impurity_decrease=0.01)
custClassTree4.fit(train_X, train_y)

#accuracy
classificationSummary(train_y, custClassTree4.predict(train_X))
classificationSummary(valid_y, custClassTree4.predict(valid_X))





#maybe this
train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.8, random_state=2)



#same aas above
custClassTree5 = DecisionTreeClassifier(max_depth=15, min_samples_split=10, min_impurity_decrease=0.01)
custClassTree5.fit(train_X, train_y)

#accuracy
classificationSummary(train_y, custClassTree5.predict(train_X))
classificationSummary(valid_y, custClassTree5.predict(valid_X))

# Confusion Matrix (Accuracy 0.8418)

#        Prediction
# Actual   0   1
#      0 264  61
#      1  70 433
# Confusion Matrix (Accuracy 0.8132)

#        Prediction
# Actual   0   1
#      0 376 103
#      1 129 634

custClassTree6 = DecisionTreeClassifier(max_depth=25, min_samples_split=15, min_impurity_decrease=0.01)
custClassTree6.fit(train_X, train_y)

#accuracy
classificationSummary(train_y, custClassTree6.predict(train_X))
classificationSummary(valid_y, custClassTree6.predict(valid_X))

# Confusion Matrix (Accuracy 0.8418)

#        Prediction
# Actual   0   1
#      0 264  61
#      1  70 433
# Confusion Matrix (Accuracy 0.8132)

#        Prediction
# Actual   0   1
#      0 376 103
#      1 129 634

train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.8, random_state=1)

# Confusion Matrix (Accuracy 0.8019)

#        Prediction
# Actual   0   1
#      0 143  29
#      1  53 189
# Confusion Matrix (Accuracy 0.7560)

#        Prediction
# Actual   0   1
#      0 518 114
#      1 290 734

train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.8, random_state=2)

# Confusion Matrix (Accuracy 0.8744)

#        Prediction
# Actual   0   1
#      0 122  33
#      1  19 240
# Confusion Matrix (Accuracy 0.8074)

#        Prediction
# Actual   0   1
#      0 444 205
#      1 114 893




custClassTree6 = DecisionTreeClassifier(max_depth=25, min_samples_split=15, min_impurity_decrease=0.000001)
custClassTree6.fit(train_X, train_y)

#accuracy
classificationSummary(train_y, custClassTree6.predict(train_X))
classificationSummary(valid_y, custClassTree6.predict(valid_X))


# Confusion Matrix (Accuracy 0.9034)

#        Prediction
# Actual   0   1
#      0 136  19
#      1  21 238
# Confusion Matrix (Accuracy 0.8128)

#        Prediction
# Actual   0   1
#      0 477 172
#      1 138 869



custClassTree6 = DecisionTreeClassifier(max_depth=20, min_samples_split=10, min_impurity_decrease=0.000001)
custClassTree6.fit(train_X, train_y)

#accuracy
classificationSummary(train_y, custClassTree6.predict(train_X))
classificationSummary(valid_y, custClassTree6.predict(valid_X))


# Confusion Matrix (Accuracy 0.9130)

#        Prediction
# Actual   0   1
#      0 146   9
#      1  27 232
# Confusion Matrix (Accuracy 0.8080)

#        Prediction
# Actual   0   1
#      0 503 146
#      1 172 835



comments_df = pd.read_excel("/Users/jbs/Downloads/Comments.xlsx")






customers_and_comments = pd.read_csv("/Users/jbs/Downloads/CustomersAndComments.csv")






















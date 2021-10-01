# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 11:58:57 2021

@author: SHRUTI
"""

# Building a decision tree model

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.model_selection import train_test_split

data = pd.read_csv(r'C:\Users\PRAKASH\Downloads\drug200.csv')
data.head()

le_Sex = LabelEncoder()
le_Bp = LabelEncoder()
le_cholesterol = LabelEncoder()
le_drug = LabelEncoder()

data['sex'] = le_Sex.fit_transform(data['Sex'])
data['bp'] = le_Bp.fit_transform(data['BP'])
data['cholesterol'] = le_cholesterol.fit_transform(data['Cholesterol'])
data['drug'] = le_drug.fit_transform(data['Drug'])

inputs = data.drop(['Sex','BP','Cholesterol','Drug'],axis='columns')

X = inputs.drop(['drug'],axis='columns')
Y = inputs['drug']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

model = tree.DecisionTreeClassifier()
model = model.fit(X_train , Y_train)

# predict the response of the dataset
y_predict = model.predict(X_test)

# Accuracy can be computed by comparing actual test set values and predicted values.
from sklearn import metrics
print("Accuracy: " , metrics.accuracy_score(Y_test,y_predict))

# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 15:15:18 2022

@author: 132855
"""

import pandas as pd
df = pd.read_csv("Social_Network_Ads.csv")


# Datalari X ve Ye bolmek
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values


# Training Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
st = StandardScaler()
X_train = st.fit_transform(X_train)
X_test = st.fit_transform(X_test)

# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
cl = DecisionTreeClassifier(criterion= "entropy", random_state=0)
cl.fit(X_train, Y_train)


# Test Model
y_pred = cl.predict(X_test)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test, y_pred)

from sklearn.metrics import accuracy_score
accuracy_score(Y_test, y_pred)

#-----------------------------------------------------------------------------------------------------

data = pd.read_csv('Position_Salaries.csv')
# Datalari X ve Ye bolmek
X = data.iloc[:, 1:-1].values
Y = data.iloc[:, -1].values


'''
# Training Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1)


# Decision Tree Regressor 
from sklearn.tree import DecisionTreeRegressor
reg = DecisionTreeRegressor(random_state=0)
reg.fit(X_train, Y_train)
'''

from sklearn.tree import DecisionTreeRegressor
reg = DecisionTreeRegressor(random_state=0)
reg.fit(X, Y)

reg.predict([[0]])





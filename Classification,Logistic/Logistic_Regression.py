# -*- coding: utf-8 -*-

import pandas as pd
df = pd.read_csv('Social_Network_Ads.csv')

df.info()


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


# Logistic Regression Model
from sklearn.linear_model import LogisticRegression
cl = LogisticRegression()
cl.fit(X_train, Y_train)

# Test Model
y_pred = cl.predict(X_test)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test, y_pred)

from sklearn.metrics import accuracy_score
accuracy_score(Y_test, y_pred)
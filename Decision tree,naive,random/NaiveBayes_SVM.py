# -*- coding: utf-8 -*-

# Merge dataframes
# https://stackoverflow.com/questions/28642177/python-pandas-dataframe-join-two-dataframes


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

# Naive Bayes
from  sklearn.naive_bayes import GaussianNB
cl = GaussianNB()
cl.fit(X_train, Y_train)

# Test Model
y_pred = cl.predict(X_test)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test, y_pred)

from sklearn.metrics import accuracy_score
accuracy_score(Y_test, y_pred)

# SVM
from sklearn.svm import SVC
cl2 = SVC(kernel = 'linear', random_state= 1)
cl2.fit(X_train, Y_train)

# Test Model
y_pred2 = cl2.predict(X_test)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test, y_pred2)

from sklearn.metrics import accuracy_score
accuracy_score(Y_test, y_pred2)




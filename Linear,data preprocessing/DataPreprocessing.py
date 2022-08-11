# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
data = pd.read_csv('Data.csv')

# data = data.fillna(data.mean())
data.fillna(data.mean(), inplace=True)

# data preprocessing: data cleaning; lemmatization, stemming; data augmentation, max pooling

# Datani bolmek
# independent
# X0 = data[:, :3]
X = data.iloc[:, :3].values
# dependent, label, target
Y = data.iloc[:, -1].values


# Scikit-Learn 
# https://scikit-learn.org/stable/
# Missing Value
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values= np.nan, strategy= 'mean')
imp.fit(X[:,1:3])
X[:,1:3] = imp.transform(X[:,1:3])

# Encoding Categorical Variables
# Y - LabelEncoding
# X - OneHotEncoding

from sklearn.preprocessing import LabelEncoder
l = LabelEncoder()
Y = l.fit_transform(Y)

# X
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
enc = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [0])], remainder= 'passthrough')
X = enc.fit_transform(X)

# Training Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
st = StandardScaler()
X_train[:,3:] = st.fit_transform(X_train[:,3:])
X_test[:,3:] = st.fit_transform(X_test[:,3:])






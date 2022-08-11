# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 16:13:39 2022
"""
# Simple Linear Regression - 1 X ile
# Multiple Linear Regression - 2 ve ya daha artiq x ile
import pandas as pd
df = pd.read_csv("Salary_Data.csv")
df2 = pd.read_csv('50_Startups.csv')

# Datalari X ve Ye bolmek
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

df.info()

# Training Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1)


# Linear Regression Modelin tetbiqi
from sklearn.linear_model import LinearRegression
rgs = LinearRegression()
rgs.fit(X_train, Y_train)


y_pred = rgs.predict(X_test)

# Visualizing Test Data
import matplotlib.pyplot as plt
plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_test, y_pred, color= 'blue')
plt.plot()

# Visualizing Train Data
import matplotlib.pyplot as plt
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, rgs.predict(X_train), color= 'blue')
plt.plot()

print(rgs.predict([[3.5]]))
print(rgs.predict([[11.5]]))

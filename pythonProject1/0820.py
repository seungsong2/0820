import pandas as pd
import numpy as np
import scipy
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

header = ['sepal - leangth', 'sepal - width', 'petal - length', 'petal - width', 'class']
data = pd.read_csv('./data/2.iris.csv', names = header)

array = data.values
X = array[:, 0:4]
Y = array[:, 4]

scaler = MinMaxScaler(feature_range=(0,1))
rescaled_X = scaler.fit_transform(X)
(X_train, X_test, Y_train, Y_test) = train_test_split(rescaled_X, Y, test_size=0.2)

model = DecisionTreeClassifier(max_depth=1000, min_samples_split=50, min_samples_leaf=5)
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)

print(confusion_matrix(y_pred, Y_test))
print(classification_report(y_pred, Y_test))
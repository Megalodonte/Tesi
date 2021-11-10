import csv
import pandas as pd
import numpy as np
from scipy.sparse import data
from sklearn.neural_network import MLPClassifier
import numpy
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

from functions import load_dataset

"""df = pd.read_csv('diabetes.csv') 
print(df.shape)
#print(df.describe().transpose())

target_column = ['Outcome'] 
predictors = list(set(list(df.columns))-set(target_column))
# df[predictors] = df[predictors]/df[predictors].max()
# print(df.describe().transpose())

X = df[predictors].values
y = df[target_column].values
y.resize((y.shape[0],))
print(X.shape); print(y.shape)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)
#print(X_train.shape); print(y_train.shape)
#y_train.resize((537,))
#print(y_train)

def get_arrays_diabetes(csv):
    df = pd.read_csv(csv)
    target_column = ['Outcome'] 
    predictors = list(set(list(df.columns))-set(target_column)) 
    X = df[predictors].values
    y = df[target_column].values
    y.resize((y.shape[0],))
    return X, y

a, b = get_arrays_diabetes("diabetes.csv")
print(a.shape)
print(b)"""

X_train, X_test, y_train, y_test, inputs, outputs, neurons_in_hidden = load_dataset(dataset_name="breast_cancer", test_size=0.3)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
MLP = MLPClassifier(hidden_layer_sizes=(neurons_in_hidden,), solver="sgd")
MLP.fit(X_train, y_train)
predicts = MLP.predict(X_test)
fitness = balanced_accuracy_score(y_test, predicts)
print("score=", fitness)



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
import os

# df = pd.read_csv("datasets/tictactoe/tictactoe.csv")
# print(df)
# print(df.isnull().sum())
for name in os.listdir("datasets"):
    print(name)


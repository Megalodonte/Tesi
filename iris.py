import operator
import random
import math
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools
import sklearn
from sklearn.neural_network import MLPClassifier
import  numpy
import matplotlib.pyplot as plt 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

MLP = MLPClassifier(max_iter=1)
MLP.fit(X_train, y_train)
"""print("Coefficienti:", MLP.coefs_)
print("Biases:", MLP.intercepts_)
print("Shape coefs", numpy.shape(MLP.coefs_))
print("Shape biases", numpy.shape(MLP.intercepts_))"""
RNA = numpy.ones(shape=(803,))
W1 = RNA[0:400].reshape((4,100))
b1 = RNA[400:500].reshape((100,))
W2 = RNA[500:800].reshape((100,3))
b2 = RNA[800:803].reshape((3,))
MLP.coefs_ = [W1,W2]
MLP.intercepts_ = [b1,b2]
print("Coefficienti:", MLP.coefs_)
print("Biases:", MLP.intercepts_)
print("Shape coefs", numpy.shape(MLP.coefs_))
print("Shape biases", numpy.shape(MLP.intercepts_))
MLP.n_layers_ = 3
fitness = MLP.score(X_train, y_train)
print("fitness = ", fitness)



from os import linesep
from scipy.sparse.construct import random
from sklearn import neural_network
from tensorflow import keras
from sklearn.neural_network import MLPClassifier
import numpy
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.datasets import load_wine
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

def get_size_prova(MLP_shape):
    mat = numpy.array([])
    bias = numpy.array([])
    sum = 0
    for i in range(len(MLP_shape) - 1):
        mat.append(MLP_shape[i:i+2])
        bias.append(MLP_shape(i+1))
    for i in mat:
        sum = sum + mat[i].size
    for i in bias:
        sum = sum + i
    return sum

def test_weights_sklearn_prova(RNA, X, y, MLP_shape):
    
    MLP_hidden = MLP_shape[1:-1]
    MLP = MLPClassifier(hidden_layer_sizes=MLP_hidden, max_iter=1)
    MLP.fit(X, y)
    RNA = numpy.array(RNA)
    mat = numpy.array([])
    bias = numpy.array([])
    for i in range(len(MLP_shape) - 1):
        mat.append(MLP_shape[i:i+2])
        bias.append(MLP_shape(i+1))
    coefs_ = numpy.array([])
    coefs_.append(numpy.reshape(RNA[:numpy.size(mat[0])], mat[0]))
    for i in range(len(MLP_shape) - 1):
        coefs_.append(numpy.reshape(RNA[numpy.size(mat[i]):numpy.size(mat[i]) + numpy.size(mat[i+1])], mat[i+1]))
        endpoint = numpy.size(mat[i]) + numpy.size(mat[i+1])
    intercepts_ = numpy.array([])
    intercepts_.append(numpy.reshape(RNA[endpoint:endpoint + bias[0]], (bias[0],)))
    for i in range(len(MLP_shape) - 1):
        intercepts_.append(numpy.reshape(RNA[endpoint + bias[i]:endpoint + bias[i] + bias[i+1]], (bias[i+1],)))

    MLP.coefs_ = coefs_
    MLP.intercepts_ = intercepts_

    fitness = MLP.score(X, y)
    return fitness,

def test_weights_keras_prova(RNA, X, y, neurons_in_hidden, inputs, outputs):
    
    #input_shape=(inputs,)
    model = keras.Sequential(
    [
        keras.layers.Dense(units=neurons_in_hidden, input_dim=inputs, activation="relu", use_bias=True, name='hidden'),
        keras.layers.Dense(units=outputs, activation="softmax", use_bias=True, name='output'),
    ]
    )
    model.compile()
    RNA = numpy.array(RNA)
    step1 = inputs*neurons_in_hidden
    step2 = step1 + neurons_in_hidden
    step3 = step2 + outputs*neurons_in_hidden
    step4 = step3 + outputs
    W1 = RNA[:step1].reshape((inputs,neurons_in_hidden))
    b1 = RNA[step1:step2].reshape((neurons_in_hidden,))
    W2 = RNA[step2:step3].reshape((neurons_in_hidden,outputs))
    b2 = RNA[step3:step4].reshape((outputs,))
    new_weights = [W1, b1, W2, b2]
    
    model.set_weights(new_weights)
    fitness = model.evaluate(X, y, verbose=0)
    return fitness,

def get_arrays_old(csv, target_column):
    df = pd.read_csv(csv)
    target_column = [target_column] 
    predictors = list(set(list(df.columns))-set(target_column)) 
    X = df[predictors].values
    y = df[target_column].values
    y.resize((y.shape[0],))
    return X, y

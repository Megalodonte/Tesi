from tensorflow import keras

from sklearn.neural_network import MLPClassifier
import numpy
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# per caricare il dataset
def load_dataset(dataset_name, test_size):

    if dataset_name == "iris": dataset = load_iris()
    elif dataset_name == "digits": dataset = load_digits()
    else: raise ValueError("il dataset non è supportato")
    
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=test_size)
    return  X_train, X_test, y_train, y_test

# per testare le particelle durante il training
def test_weights_sklearn(RNA, X, y, neurons_in_hidden, inputs, outputs):
    
    MLP = MLPClassifier(max_iter=1)
    MLP.fit(X, y)
    RNA = numpy.array(RNA)
    step1 = inputs*neurons_in_hidden
    step2 = step1 + neurons_in_hidden
    step3 = step2 + outputs*neurons_in_hidden
    step4 = step3 + outputs
    W1 = RNA[:step1].reshape((inputs,neurons_in_hidden))
    b1 = RNA[step1:step2].reshape((neurons_in_hidden,))
    W2 = RNA[step2:step3].reshape((neurons_in_hidden,outputs))
    b2 = RNA[step3:step4].reshape((outputs,))
    MLP.coefs_ = [W1,W2]
    MLP.intercepts_ = [b1,b2]
    fitness = MLP.score(X, y)
    return fitness,

def test_weights_keras(RNA, X, y, neurons_in_hidden, inputs, outputs):
    
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

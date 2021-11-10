import sklearn
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPClassifier
import numpy
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.datasets import load_wine
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# per caricare il dataset
def load_dataset(dataset_name, test_size):

    class ds:
        def __init__(self, data, target):
            self.data = data
            self.target = target

    if dataset_name == "iris": 
        dataset = load_iris()
        inputs = 4
        outputs = 3
    
    elif dataset_name == "digits": 
        dataset = load_digits()
        inputs = 64
        outputs = 10   
    
    elif dataset_name == "wine":
        dataset = load_wine()
        inputs = 13
        outputs = 3 
    
    elif dataset_name == "breast_cancer":
        dataset = load_breast_cancer()
        inputs = 30
        outputs = 1

    elif dataset_name == "diabetes":
        d, t = get_arrays_diabetes("diabetes.csv")
        dataset = ds(data=d, target=t)
        inputs = 8
        outputs = 1
    
    else: raise ValueError("il dataset non è supportato")
    
    neurons = inputs*2 + 1
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=test_size, random_state=40)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return  X_train, X_test, y_train, y_test, inputs, outputs, neurons

# per testare le particelle durante il training
def test_weights_sklearn(RNA, X, y, inputs, outputs, neurons_in_hidden, id=bool):
    
    MLP = MLPClassifier(hidden_layer_sizes=(neurons_in_hidden,), max_iter=1)
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
    predicts = MLP.predict(X)   
    fitness = balanced_accuracy_score(y, predicts)
    return fitness,

def get_size(inputs, outputs, neurons):
    size = (inputs+outputs)*neurons + outputs + neurons
    return size

def get_arrays_diabetes(csv):
    df = pd.read_csv(csv)
    target_column = ['Outcome'] 
    predictors = list(set(list(df.columns))-set(target_column)) 
    X = df[predictors].values
    y = df[target_column].values
    y.resize((y.shape[0],))
    return X, y
import numpy
import json
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.datasets import load_wine
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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

    elif dataset_name == "breast_cancer_sklearn":
        dataset = load_breast_cancer()
        inputs = 30
        outputs = 1

    elif dataset_name == "diabetes":
        d, t = get_arrays("datasets/diabetes/diabetes.csv", "Outcome")
        dataset = ds(data=d, target=t)
        inputs = 8
        outputs = 1

    elif dataset_name == "blood":
        d, t = get_arrays("datasets/blood/transfusion.data", "whether he/she donated blood in March 2007")
        dataset = ds(data=d, target=t)
        inputs = 4
        outputs = 1

    elif dataset_name == "parkinson":
        d, t = get_arrays("datasets/parkinson/parkinsons.data", "status", first_column=True)
        dataset = ds(data=d, target=t)
        inputs = 22
        outputs = 1

    elif dataset_name == "breast_cancer":
        d, t = get_arrays("datasets/breast/breast-cancer-wisconsin.data", "Column10", first_column=True, first_row=True)
        dataset = ds(data=d, target=t)
        inputs = 9
        outputs = 1

    elif dataset_name == "vertebral":
        d, t = get_arrays("datasets/vertebral/column_3C.dat", "Column6", first_row=True, sep=" ")
        dataset = ds(data=d, target=t)
        inputs = 6
        outputs = 3

    elif dataset_name == "seeds":
        d, t = get_arrays("datasets/seeds/seeds_dataset.txt", "Column7", first_row=True, sep=" ")
        dataset = ds(data=d, target=t)
        inputs = 7
        outputs = 3

    elif dataset_name == "balance":
        d, t = get_arrays("datasets/balance/balance-scale.data", "Column4", first_row=True)
        dataset = ds(data=d, target=t)
        inputs = 4
        outputs = 3

    elif dataset_name == "vehicle":
        d, t = get_arrays("datasets/vehicle/vehicle.dat", "Column18", first_row=True, sep=" ")
        dataset = ds(data=d, target=t)
        inputs = 18
        outputs = 4

    elif dataset_name == "glass":
        d, t = get_arrays("datasets/glass/glass.csv", "Type")
        dataset = ds(data=d, target=t)
        inputs = 9
        outputs = 7

    elif dataset_name == "ph":
        d, t = get_arrays("datasets/ph/ph-data.csv", "label")
        dataset = ds(data=d, target=t)
        inputs = 3
        outputs = 15

    elif dataset_name == "alcohol":
        d, t = get_arrays("datasets/alcohol/QCM3.csv", "class", sep=";")
        dataset = ds(data=d, target=t)
        inputs = 10
        outputs = 5

    elif dataset_name == "teaching":
        d, t = get_arrays("datasets/teaching/teaching.csv", "class", first_column=True)
        dataset = ds(data=d, target=t)
        inputs = 5
        outputs = 3

    elif dataset_name == "car":
        d, t = get_arrays("datasets/car/car.csv", "class")
        dataset = ds(data=d, target=t)
        inputs = 6
        outputs = 4

    elif dataset_name == "penguins":
        d, t = get_arrays("datasets/penguins/penguins.csv", "species")
        dataset = ds(data=d, target=t)
        inputs = 6
        outputs = 3

    elif dataset_name == "hayes":
        d, t = get_arrays("datasets/hayes/hayes.csv", "class")
        dataset = ds(data=d, target=t)
        inputs = 4
        outputs = 3

    elif dataset_name == "nursery":
        d, t = get_arrays("datasets/nursery/nursery.csv", "Class")
        dataset = ds(data=d, target=t)
        inputs = 8
        outputs = 4

    else: raise ValueError("il dataset non è supportato")

    neurons = inputs*2 + 1
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=test_size, random_state=69)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return  X_train, X_test, y_train, y_test, inputs, outputs, neurons

# per testare le particelle durante il training
def test_weights_sklearn(RNA, X, y, inputs, outputs, neurons_in_hidden, id=True):

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
    if id:
        fitness = balanced_accuracy_score(y, predicts)
    else:
        fitness = accuracy_score(y, predicts)
    return fitness,

def get_size(inputs, outputs, neurons):
    size = (inputs+outputs)*neurons + outputs + neurons
    return size

def get_arrays(csv, target_column, first_column=False, first_row=False, sep=","):
    if first_row:
        if first_column:
            df = pd.read_csv(csv, header=None, index_col=[0], prefix="Column", sep=sep)
        else:
            df = pd.read_csv(csv, header=None, prefix="Column", sep=sep)
    else:
        if first_column:
            df = pd.read_csv(csv, index_col=[0], sep=sep)
        else:
            df = pd.read_csv(csv, sep=sep)
    dfcopy = df.copy()
    dfcopy.drop(target_column, 1, inplace=True)
    X = dfcopy.values
    target = [target_column]
    y = df[target].values
    y.resize((y.shape[0],))
    return X, y

def save_test(name, train, test, wb, log):
    trainfile = "data_balanced_accuracy/%s/%s_train_vector.json" %(name, name)
    with open(trainfile, "w") as f:
        json.dump(train, f)
    testfile = "data_balanced_accuracy/%s/%s_test_vector.json" %(name, name)
    with open(testfile, "w") as f:
        json.dump(test, f)
    wbfile = "data_balanced_accuracy/%s/%s_w&b.json" %(name, name)
    with open(wbfile, "w") as f:
        json.dump(wb, f)
    logfile = "data_balanced_accuracy/%s/%s_logbooks.json" %(name, name)
    with open(logfile, "w") as f:
        json.dump(log, f)

def save_test_BP(name, train, test):
    trainfile = "data_BP/%s/%s_train_vector.json" %(name, name)
    with open(trainfile, "w") as f:
        json.dump(train, f)
    testfile = "data_BP/%s/%s_test_vector.json" %(name, name)
    with open(testfile, "w") as f:
        json.dump(test, f)

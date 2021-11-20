import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import functions
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
import numpy as np

simplefilter("ignore", category=ConvergenceWarning)

dataset_name = "liver"
test_size = 0.3
X_train, X_test, y_train, y_test, num_inputs, num_outputs, neurons_in_hidden = functions.load_dataset(dataset_name=dataset_name, test_size=test_size)
num_tests = 15
train_vector = []
test_vector = []
for i in range(num_tests):
    print("--------Test %d/%d--------" %(i+1,num_tests))
    model = MLPClassifier(hidden_layer_sizes=(neurons_in_hidden,), solver="sgd")
    model.fit(X_train, y_train)
    train_predicts = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_predicts)
    print("Score sul train set =", train_acc)
    train_vector.append(train_acc)
    test_predicts = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_predicts)
    print("Accuracy sul test set =", test_acc)
    test_vector.append(test_acc)

functions.save_test_BP(dataset_name, train_vector, test_vector)
print("--------Fine dei test--------")


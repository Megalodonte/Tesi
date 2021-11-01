import  numpy
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.engine.input_layer import Input
from tensorflow.keras import Model
dataset_name = "iris"
test_size = 0.3
phi1 = 2.0
phi2 = 2.0
neurons_in_hidden = 100
pmin = -100
pmax = 100
smin = -5
smax = 5
num_inputs = 4
num_outputs = 3
size = (num_inputs+num_outputs)*neurons_in_hidden + num_outputs + neurons_in_hidden
size_pop = 100
generations = 100
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

input_shape=(num_inputs,)
"""model = Sequential()
model.add(Input(shape=input_shape))
model.add(Dense(neurons_in_hidden, activation="sigmoid"))
model.add(Dense(num_outputs, activation="softmax"))"""
inputs = Input(shape=(4,))                    # input layer
x = Dense(100, activation='relu')(inputs)     # hidden layer
outputs = Dense(3, activation='softmax')(x)  # output layer

model = Model(inputs, outputs)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=50, verbose=1, validation_split=0.2)
weights = model.get_weights()
print("Matrici:", weights)
for i in range(len(weights)):
    print("Shape[%d]: "%i)
    print((numpy.shape(weights[i])))
fitness_init = model.evaluate(X_train, y_train, verbose=0)
print("fitness iniziale = ", fitness_init)
RNA = numpy.ones(shape=(803,))
W1 = RNA[0:400].reshape((4,100))
b1 = RNA[400:500].reshape((100,))
W2 = RNA[500:800].reshape((100,3))
b2 = RNA[800:803].reshape((3,))
new_weights = [W1, b1, W2, b2]
model.set_weights(new_weights)
modified_weights = model.get_weights()
print("Matrici:", modified_weights)
for i in range(len(modified_weights)):
    print("Shape[%d]: "%i)
    print((numpy.shape(modified_weights[i])))
fitness_fin = model.evaluate(X_train, y_train, verbose=0)
print("fitness finale = ", fitness_fin)




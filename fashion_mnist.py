import numpy as np
from util import *
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.optimizers import SGD

# Get Fashion MNIST dataset
'''
Original split:
    Train: 60000 samples
        0: 6000 samples
        1: 6000 samples
        2: 6000 samples
        3: 6000 samples
        4: 6000 samples
        5: 6000 samples
        6: 6000 samples
        7: 6000 samples
        8: 6000 samples
        9: 6000 samples
    Test: 10000 samples
        0: 1000 samples
        1: 1000 samples
        2: 1000 samples
        3: 1000 samples
        4: 1000 samples
        5: 1000 samples
        6: 1000 samples
        7: 1000 samples
        8: 1000 samples
        9: 1000 samples
'''
train_x, train_y, test_x, test_y = load_keras_dataset(keras.datasets.fashion_mnist)




# Leave out 5 of the classes during training. Classes 5-9 are designated as "unkown"
known_x, known_y = [], []
unknown_x, unknown_y = [], []
for i in range(train_x.shape[0]):
    if train_y[i] >= 5:
        unknown_x.append(train_x[i])
        unknown_y.append(train_y[i])
    else:
        known_x.append(train_x[i])
        known_y.append(train_y[i])

train_x = np.array(known_x)
train_y = np.array(known_y)
test_x = np.concatenate((test_x, np.array(unknown_x)), axis=0)
test_y = np.concatenate((test_y, np.array(unknown_y)), axis=0)

# Change labels of unkown classes to be negative numbers
for i in range(len(test_y)):
    if test_y[i] >= 5:
        test_y[i] = -test_y[i] + 4
"""
New split:
    Train: 30000 samples
        0: 6000 samples
        1: 6000 samples
        2: 6000 samples
        3: 6000 samples
        4: 6000 samples
    Test: 40000 samples
        0: 1000 samples
        1: 1000 samples
        2: 1000 samples
        3: 1000 samples
        4: 1000 samples
        -1: 7000 samples
        -2: 7000 samples
        -3: 7000 samples
        -4: 7000 samples
        -5: 7000 samples
"""
print()




# preprocess & normalization
train_x = train_x / 255.0
test_x = test_x / 255.0




print("\nTraining/Loading model")
model_path = './saved_models/fashion_mnist.h5'
train = False

model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(128, activation=tf.nn.relu))
model.add(keras.layers.Dense(32, activation=tf.nn.relu))
model.add(keras.layers.Dense(5, activation=tf.nn.softmax))
model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

if train:
    model.fit(train_x, train_y, epochs=10)
    model.save(model_path)
else:
    model = keras.models.load_model(model_path)


print("\nTesting model")


threshold_to_test = np.linspace(0, 1, 50)
perform_analysis(model, test_x, test_y, threshold_to_test, 'fashion_mnist', 0.8)

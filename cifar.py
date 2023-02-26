import numpy as np
from util import *
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.optimizers import SGD
from tensorflow.keras.layers import *
from sklearn.metrics import confusion_matrix
import seaborn
import tensorflow_datasets as tfds
from tensorflow.python.ops.numpy_ops import np_config
from scipy.ndimage.filters import gaussian_filter
np_config.enable_numpy_behavior()











# Get CIFAR-100 dataset
'''
Train: 50000 samples
    labels: 0-99
    500 samples per label
Test: 10000 samples
    labels: 0-99
    around 100 samples per label

Each image is 32x32 in color
'''
train_x, train_y, test_x, test_y = load_keras_dataset(keras.datasets.cifar100)








# Leave out half (50) of the classes during training. Classes 50-99 are designated as "unkown"
known_x, known_y = [], []
unknown_x, unknown_y = [], []
for i in range(train_x.shape[0]):
    if train_y[i] >= 50:
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
    if test_y[i] >= 50:
        test_y[i] = -1
"""
New split:
    Train: 25000 samples
        labels: 0-49
        500 samples per label
    Test: 35000 samples
        labels: 0-49 have 100 samples per label
        label -1 has 30000 samples
"""





exit()










print("\nTraining model")
model = keras.Sequential()
model.add(keras.layers.Conv2D(filters=4, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(train_x[0].shape[0], train_x[0].shape[1], 1)))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(32, activation=tf.nn.relu))
model.add(keras.layers.Dense(10, activation=tf.nn.softmax))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_x, train_y, epochs=5)








print("\nTesting model")
threshold_to_test = np.linspace(0, 1, 200)
perform_analysis(model, test_x, test_y, threshold_to_test, 'mnist', 0.9)





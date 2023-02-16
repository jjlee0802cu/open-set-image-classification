import numpy as np
import tensorflow as tf
from tensorflow import keras






# Get Fashion MNIST dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()





# Leave out 5 of the classes during training. Classes 5-9 are designated as "unkown"
"""
Resulting split:
    Train:
        0-4 have 6000 samples
    Test:
        0-4 have 1000 samples
        5-9 have 7000 samples

"""
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

print(train_x.shape, train_y.shape)
print(test_x.shape, test_y.shape)





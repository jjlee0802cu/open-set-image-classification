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














# Get MNIST dataset
'''
Train: 60000 samples
    labels: 0-9
    around 6000 samples per label
Test: 10000 samples
    labels: 0-9
    around 1000 samples per label
'''
train_x, train_y, test_x, test_y = load_keras_dataset(keras.datasets.mnist)


# Pre-process MNIST dataset
'''
make grayscale
'''
train_x = train_x / 255.0
test_x = test_x / 255.0
















# Get Omniglot dataset (combine omniglot train and test into 1 omniglot dataset since they are all going to be unkown samples)
'''
32,460 samples in total
1623 labels with 20 samples per label
'''
omniglot_train_x, omniglot_train_y = load_omniglot('train')
omniglot_test_x, omniglot_test_y = load_omniglot('test')
omniglot_x = omniglot_train_x + omniglot_test_x
omniglot_y = omniglot_train_y + omniglot_test_y


# Pre-process omniglot images
'''
resize to 28x28 (same size as mnist images)
make make values be between 0 and 1 just like mnist was normalized
flip pixels so that it's black background, white foreground
add a little bit of gaussian filter to blur it
'''
omniglot_x = [(gaussian_filter(1.0 - (tf.image.resize(i, [28, 28])/255.0), sigma=0.5))[:, :, 0] for i in omniglot_x]

# Change labels of unkown classes to be negative numbers
for i in range(len(omniglot_y)):
    omniglot_y[i] = -1




















# Make final train/test sets
'''
Train: 60000 samples
    labels: 0-9
    around 6000 samples per label
Test: 42,460 samples
    labels: -1623~9
        labels -1623~-1 have 20 samples per label
        labels 0~9 have approx 1000 samples per label 
'''
omniglot_x = np.array(omniglot_x)
omniglot_y = np.array(omniglot_y)
test_x = np.concatenate((test_x, omniglot_x), axis=0)
test_y = np.concatenate((test_y, omniglot_y), axis=0)





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





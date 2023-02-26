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
# Details: 
# - https://huggingface.co/datasets/cifar100
# - https://www.tensorflow.org/datasets/catalog/cifar100
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







# preprocess & normalization
train_x = train_x / 255.0
test_x = test_x / 255.0






print("\nTraining/Loading model")
model_path = './saved_models/cifar_cnn.h5'
train = False

model = keras.Sequential()
model.add(keras.layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=train_x[0].shape))
model.add(keras.layers.Conv2D(64, 3, padding='same', activation='relu'))
model.add(keras.layers.Conv2D(128, 3, padding='same', activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation=tf.nn.relu))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(128, activation=tf.nn.relu))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(50, activation=tf.nn.softmax))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

if train:
    model.fit(train_x, train_y, epochs=20, batch_size=128)
    model.save(model_path)
else:
    model = keras.models.load_model(model_path)



exit()

print("\nTesting model")
threshold_to_test = np.linspace(0, 1, 200)
perform_analysis(model, test_x, test_y, threshold_to_test, 'cifar', 0.9)


'''

Epoch 1/10
196/196 [==============================] - 31s 158ms/step - loss: 3.3719 - accuracy: 0.1673
Epoch 2/10
196/196 [==============================] - 30s 155ms/step - loss: 2.6012 - accuracy: 0.3136
Epoch 3/10
196/196 [==============================] - 31s 158ms/step - loss: 2.0985 - accuracy: 0.4257
Epoch 4/10
196/196 [==============================] - 31s 156ms/step - loss: 1.6988 - accuracy: 0.5236
Epoch 5/10
196/196 [==============================] - 30s 155ms/step - loss: 1.3383 - accuracy: 0.6232
Epoch 6/10
196/196 [==============================] - 30s 155ms/step - loss: 0.9365 - accuracy: 0.7313
Epoch 7/10
196/196 [==============================] - 30s 153ms/step - loss: 0.5375 - accuracy: 0.8526
Epoch 8/10
196/196 [==============================] - 30s 152ms/step - loss: 0.2891 - accuracy: 0.9258
Epoch 9/10
196/196 [==============================] - 30s 152ms/step - loss: 0.1443 - accuracy: 0.9688
Epoch 10/10
196/196 [==============================] - 31s 161ms/step - loss: 0.0869 - accuracy: 0.9832

'''



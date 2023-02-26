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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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

# Reformat dimensions for the labels (train_y and test_y)
train_y = np.squeeze(train_y, axis=1)
test_y = np.squeeze(test_y, axis=1)
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
train = True

model = keras.Sequential()
model.add(keras.layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=train_x[0].shape))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(64, 3, padding='same', activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(128, 3, padding='same', activation='relu'))
model.add(keras.layers.BatchNormalization())
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
    # Create validation set (which is the test set without the -1 labels)
    val_x = np.array([test_x[i] for i in range(len(test_x)) if test_y[i] >= 0])
    val_y = np.array([test_y[i] for i in range(len(test_x)) if test_y[i] >= 0])
    # Create data augmeter using ImageDataGenerator
    augmeter = ImageDataGenerator(rotation_range=20, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    augmeter.fit(train_x)
    # Fit and save (uses real-time augmeter which is generating batches of size 128 of augmented data)
    model.fit(augmeter.flow(train_x, train_y, batch_size=128), validation_data=(val_x, val_y), batch_size=128, epochs=50)
    model.save(model_path)
else:
    model = keras.models.load_model(model_path)



exit()


print("\nTesting model")
threshold_to_test = np.linspace(0, 1, 200)
# perform_analysis(model, test_x, test_y, threshold_to_test, 'cifar', 0.9, False)






correct,total = 0,0
predictions = model.predict(test_x)
for i in range(len(test_y)):
    a = np.argmax(predictions[i])
    b = test_y[i]

    if b >= 0:
        total += 1
        if a == b:
            correct += 1
print(correct/total)

import numpy as np
import tensorflow as tf
from tensorflow import keras

# Get Fashion MNIST dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()

# Convert all to float types
train_x = train_x.astype(float)
train_y = train_y.astype(float)
test_x = test_x.astype(float)
test_y = test_y.astype(float)

# Leave out 5 of the classes during training. Classes 5-9 are designated as "unkown"
"""
Original split:
    Train: 60000 samples
        0~9 have 6000 samples each
    Test: 10000 samples
        0~9 have 1000 samples each

Resulting split:
    Train: 30000 samples
        0~4 have 6000 samples each
    Test: 40000 samples
        0~4 have 1000 samples each
        -5~-9 has 7000 samples each
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

# Change labels of unkown classes to be negative numbers
for i in range(len(test_y)):
    if test_y[i] >= 5:
        test_y[i] = -test_y[i] + 4




# normalization
train_x = train_x / 255.0
test_x = test_x / 255.0

# this one uses cnn: https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-fashion-mnist-clothing-classification/

# this is baseline model from https://www.kaggle.com/code/arunkumarramanan/awesome-cv-with-fashion-mnist-classification/notebook
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_x, train_y, epochs=5)






# Testing
predictions = model.predict(test_x)


correct, total = 1, 1
for i in range(predictions.shape[0]):
    softmax_output = predictions[i]

    y_pred = np.argmax(softmax_output)
    y_test = test_y[i]

    if y_test >= 0:
        if y_pred == y_test:
            correct += 1
        total += 1

print(correct/total)
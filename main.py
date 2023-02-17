import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import DetCurveDisplay


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
fashion_mnist = keras.datasets.fashion_mnist
(train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()

# Convert all to float types
train_x = train_x.astype(float)
train_y = train_y.astype(float)
test_x = test_x.astype(float)
test_y = test_y.astype(float)

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
        test_y[i] = -1
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
        -1: 35000 samples
"""
print()





print("\nTraining model")
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





print("\nTesting model")
# Testing

def sign(num):
    return 1 if num >= 0 else -1

def apply_threshold(threshold, predictions):
    threshold_predictions = []
    for i in range(predictions.shape[0]):
        argmax = np.argmax(predictions[i])
        confidence = np.max(predictions[i])
        if confidence >= threshold:
            threshold_predictions.append(argmax)
        else:
            threshold_predictions.append(-1)
    return threshold_predictions

predictions = model.predict(test_x)
threshold_predictions = apply_threshold(0.9, predictions)


correct = 0
c,t = 0,0
for i in range(len(threshold_predictions)):
    if sign(threshold_predictions[i]) == sign(test_y[i]):
        correct += 1
    if sign(threshold_predictions[i]) > 0 and sign(test_y[i]) > 0:
        t += 1
        if threshold_predictions[i] == test_y[i]:
            c += 1
print(correct/len(threshold_predictions))
print(c/t)




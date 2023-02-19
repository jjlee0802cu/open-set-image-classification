import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import DetCurveDisplay
import matplotlib.pyplot as plt

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
    try:
        return 1 if num >= 0 else -1
    except:
        return 1 if min(num) >= 0 else -1

def apply_threshold_top_N(threshold, predictions, N):
    threshold_predictions = []
    for i in range(predictions.shape[0]):
        confidence = np.max(predictions[i])
        if confidence >= threshold:
            threshold_predictions.append(set(np.argsort(predictions[i])[::-1][:N]))
        else:
            threshold_predictions.append(-1)
    return threshold_predictions

def get_accuracy(dictionary):
    try:
        return dictionary['correct']/dictionary['total']
    except:
        return 1




predictions = model.predict(test_x)


tpr_list = []
fpr_list = []
fnr_list = []
tnr_list = []
n = 5
id_accuracies = [[] for _ in range(n)]



threshold_to_test = np.linspace(0, 1, 50)
for threshold in threshold_to_test:
    print("threshold:", threshold)

    top_N_predictions = [apply_threshold_top_N(threshold, predictions, i) for i in range(1,n+1)]
    top_N_id_accuracy = [{'correct': 0, 'total': 0} for _ in range(len(top_N_predictions))]

    '''
    Definitions
        positive: known
        negative: unkown

        For example, false positive: predicted positive but it's wrong, so that means you predicted a sammple from unknown class as known
    '''
    tpr = {'correct': 0, 'total': 0}
    fpr = {'correct': 0, 'total': 0}
    fnr = {'correct': 0, 'total': 0}
    tnr = {'correct': 0, 'total': 0}
    
    for i in range(len(predictions)):
        # Get the 4 rates
        if sign(test_y[i]) >= 0: # positives
            tpr['total'] += 1
            fnr['total'] += 1
            if sign(top_N_predictions[0][i]) >= 0: # true positive
                tpr['correct'] += 1
            else: # false negative
                fnr['correct'] += 1
        else: # negatives
            fpr['total'] += 1
            tnr['total'] += 1
            if sign(top_N_predictions[0][i]) >= 0: # false positive
                fpr['correct'] += 1
            else: # true negative
                tnr['correct'] += 1

        # Top-N accuracies
        for j in range(len(top_N_predictions)):
            if sign(top_N_predictions[0][i]) >= 0 and sign(test_y[i]) >= 0:
                top_N_id_accuracy[j]['total'] += 1
                if test_y[i] in top_N_predictions[j][i]:
                    top_N_id_accuracy[j]['correct'] += 1

    tpr_list.append(get_accuracy(tpr))
    fpr_list.append(get_accuracy(fpr))
    fnr_list.append(get_accuracy(fnr))
    tnr_list.append(get_accuracy(tnr))
    for k in range(len(top_N_id_accuracy)):
        id_accuracies[k].append(get_accuracy(top_N_id_accuracy[k]))



plt.clf()
plt.plot(threshold_to_test, id_accuracies[0])
plt.xlabel('Threshold')
plt.ylabel('Accuracy')
plt.savefig('./plots/fashion_mnist/id_accuracy.png')

plt.clf()
for i in range(len(id_accuracies)):
    plt.plot(threshold_to_test, id_accuracies[i], label='Top-'+str(i+1))
plt.legend(loc="best")
plt.xlabel('Threshold')
plt.ylabel('Accuracy')
plt.savefig('./plots/fashion_mnist/top_N_id_accuracy.png')

plt.clf()
plt.plot(threshold_to_test,  tpr_list, label='TPR')
plt.plot(threshold_to_test,  fpr_list, label='FPR')
plt.plot(threshold_to_test,  fnr_list, label='FNR')
plt.plot(threshold_to_test,  tnr_list, label='TNR')
plt.legend(loc="best")
plt.axis([-0.05, 1.05, -0.05, 1.05])
plt.xlabel('Threshold')
plt.ylabel('Rates')
plt.savefig('./plots/fashion_mnist/all_rates.png')

plt.clf()
plt.plot(fpr_list, tpr_list)
plt.axis([-0.05, 1.05, -0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig('./plots/fashion_mnist/roc.png')

plt.clf()
plt.plot(fpr_list, fnr_list)
plt.axis([-0.05, 1.05, -0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('False Negative Rate')
plt.savefig('./plots/fashion_mnist/det.png')







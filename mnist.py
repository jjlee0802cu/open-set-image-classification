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
make grayscale
flip pixels so that it's black background, white foreground
'''
omniglot_x = [(1.0 - tf.image.resize(tf.image.rgb_to_grayscale(i), [28, 28]))[:, :, 0] for i in omniglot_x]


# Change labels of unkown classes to be negative numbers
for i in range(len(omniglot_y)):
    omniglot_y[i] = -omniglot_y[i] -1















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
model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(train_x[0].shape[0], train_x[0].shape[1], 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation=tf.nn.relu))
model.add(Dense(64, activation=tf.nn.relu))
model.add(Dense(10, activation=tf.nn.softmax))
model.compile(optimizer=SGD(lr=0.01, momentum=0.9), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_x, train_y, epochs=10, batch_size=64)




exit()




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
plt.savefig('./plots/mnist/id_accuracy.png')

plt.clf()
for i in range(len(id_accuracies)):
    plt.plot(threshold_to_test, id_accuracies[i], label='Top-'+str(i+1))
plt.legend(loc="best")
plt.xlabel('Threshold')
plt.ylabel('Accuracy')
plt.savefig('./plots/mnist/top_N_id_accuracy.png')

plt.clf()
plt.plot(threshold_to_test,  tpr_list, label='TPR')
plt.plot(threshold_to_test,  fpr_list, label='FPR')
plt.plot(threshold_to_test,  fnr_list, label='FNR')
plt.plot(threshold_to_test,  tnr_list, label='TNR')
plt.legend(loc="best")
plt.axis([-0.05, 1.05, -0.05, 1.05])
plt.xlabel('Threshold')
plt.ylabel('Rates')
plt.savefig('./plots/mnist/all_rates.png')

plt.clf()
plt.plot(fpr_list, tpr_list)
plt.axis([-0.05, 1.05, -0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig('./plots/mnist/roc.png')

plt.clf()
plt.plot(fpr_list, fnr_list)
plt.axis([-0.05, 1.05, -0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('False Negative Rate')
plt.savefig('./plots/mnist/det.png')


cm_test, cm_pred = [], []
best_threshold_predictions = apply_threshold_top_N(0.8, predictions, 1)
for i in range(len(best_threshold_predictions)):
    if sign(best_threshold_predictions[i]) >= 0 and sign(test_y[i]) >= 0:
        cm_test.append(test_y[i])
        cm_pred.append(min(best_threshold_predictions[i]))
cm = confusion_matrix(cm_test, cm_pred)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, ax = plt.subplots(figsize=(10,10))
seaborn.heatmap(cm, cmap="Blues", annot=True, fmt='.2f')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('./plots/mnist/confusion_matrix.png')

'''
The confusion matrix shows that class 4 is hardest to predict: Coat
It confuses class 4 with class 2, which is Pullover
'''

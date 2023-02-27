import numpy as np
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn

def load_keras_dataset(dataset):
    (train_x, train_y), (test_x, test_y) = dataset.load_data()

    # Convert all to float types
    train_x = train_x.astype(float)
    train_y = train_y.astype(float)
    test_x = test_x.astype(float)
    test_y = test_y.astype(float)

    return train_x, train_y, test_x, test_y

def print_counts_from_numpy(x):
    # prints the counts of each unique item in a numpy array (for debugging)
    elements, frequency = np.unique(x, return_counts=True)
    print(np.asarray((elements, frequency)).T)


def load_omniglot(split='train'):
    # Loads the omniglot dataset from tensorflow
    omniglot_x, omniglot_y = [], []
    ds = tfds.load("omniglot", split=split, as_supervised=True)
    for image, label in ds:
        omniglot_x.append(image)
        omniglot_y.append(label.numpy())
    return omniglot_x, omniglot_y

def sign(num):
    # returns 1 if input is non-negative. Returns -1 otherwise
    try:
        return 1 if num >= 0 else -1
    except:
        return 1 if min(num) >= 0 else -1

def apply_threshold_top_N(threshold, predictions, N):
    # Given a list or softmax outputs, apply a threshold to it
    # If the argmax of softmax output is greater or equal to threshold, output that label
    # Otherwise, output -1 (unknown)

    threshold_predictions = []
    for i in range(predictions.shape[0]):
        confidence = np.max(predictions[i])
        if confidence >= threshold:
            threshold_predictions.append(set(np.argsort(predictions[i])[::-1][:N]))
        else:
            threshold_predictions.append(-1)
    return threshold_predictions

def get_accuracy(dictionary):
    # Given a dictionary with correct and total keys, return correct/total
    try:
        return dictionary['correct']/dictionary['total']
    except:
        return 1

def perform_analysis(model, test_x, test_y, threshold_to_test, output_dir, cf_threshold, cf_annot=True):
    # Performs analysis and saves graphs to output directory
    
    predictions = model.predict(test_x)

    tpr_list = []
    fpr_list = []
    fnr_list = []
    tnr_list = []
    n = 5
    id_accuracies = [[] for _ in range(n)]

    for threshold in threshold_to_test:
        print("threshold:", threshold)

        # Get top n predictions
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

    # Plot of top-1 accuracy
    plt.clf()
    plt.plot(threshold_to_test, id_accuracies[0])
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.savefig('./plots/'+output_dir+'/id_accuracy.png')

    # Plot of top-n accuracies in one graph
    plt.clf()
    for i in range(len(id_accuracies)):
        plt.plot(threshold_to_test, id_accuracies[i], label='Top-'+str(i+1))
    plt.legend(loc="best")
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.savefig('./plots/'+output_dir+'/top_N_id_accuracy.png')

    # Plot tpr, fpr, fnr, tnr in one graph
    plt.clf()
    plt.plot(threshold_to_test,  tpr_list, label='TPR')
    plt.plot(threshold_to_test,  fpr_list, label='FPR')
    plt.plot(threshold_to_test,  fnr_list, label='FNR')
    plt.plot(threshold_to_test,  tnr_list, label='TNR')
    plt.legend(loc="best")
    plt.axis([-0.05, 1.05, -0.05, 1.05])
    plt.xlabel('Threshold')
    plt.ylabel('Rates')
    plt.savefig('./plots/'+output_dir+'/all_rates.png')

    # Plot an ROC curve
    plt.clf()
    plt.plot(fpr_list, tpr_list)
    plt.axis([-0.05, 1.05, -0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig('./plots/'+output_dir+'/roc.png')

    # Plot a DET curve
    plt.clf()
    plt.axis('scaled')
    plt.plot(fpr_list, fnr_list)
    plt.plot([0,1], [0,1], linestyle='dashed')
    plt.axis([-0.05, 1.05, -0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('False Negative Rate')
    plt.savefig('./plots/'+output_dir+'/det.png')

    # Make a confusion matrix using seaborn
    cm_test, cm_pred = [], []
    best_threshold_predictions = apply_threshold_top_N(cf_threshold, predictions, 1)
    for i in range(len(best_threshold_predictions)):
        if sign(best_threshold_predictions[i]) >= 0 and sign(test_y[i]) >= 0:
            cm_test.append(test_y[i])
            cm_pred.append(min(best_threshold_predictions[i]))
    cm = confusion_matrix(cm_test, cm_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(10,10))
    seaborn.heatmap(cm, cmap="Blues", annot=cf_annot, fmt='.2f')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('./plots/'+output_dir+'/confusion_matrix.png')
    

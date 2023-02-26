# Open-Set Image Classification

## Author

Justin Lee (UNI: jjl2245)

## Project Summary

Classification tasks require a model to learn how to take an input and assign it a label. Most classification tasks are closed-set classification tasks; in other words, the model assumes that the test data contains the same set of labels as the train data. The major shortcoming of this approach is real-world applicability: in many real-world problems, we may never be able to foresee and exhaust all possible classes during training. Open-set classifiers, on the other hand, acknowledge an incomplete knowledge of the label space and are classifiers that can reject the input and label it as “unknown” rather than assigning it an incorrect label. 

The goal of this project was to tackle the open-set image classification problem by building open-set image classification models that can not only accurately classify images from known classes, but also accurately reject images from unknown classes. 

<details><summary>Approach</summary>
<p>

- The methodology implemented and tested in this project is to first build a neural net to classify the known classes with a softmax output as its final layer. Then, use a constant confidence threshold function to either accept the prediction or reject it and output “unknown.” The way the threshold function works would be to look at the softmax output, and if none of the labels have confidence greater or equal to the threshold, reject the prediction and label it as “unknown.”

</p>
</details>

<details><summary>Applications and Datasets</summary>
<p>

- In this project, I implemented the above approach and analyzde the feasibility and performance of the approach in 3 different applications:
    - Application 1: Clothing image classification
        - Fashion MNIST dataset contains 28x28 grayscale images of fashion items coming from 10 classes. To turn this into an open-set image classification problem, 5 of the classes will be left out during training and designated as “unknown.” The model will see the other 5 classes during training. 
    - Application 2: Handwritten characters classification
        - MNIST dataset contains 28x28 grayscale images of handwritten digits coming from 10 classes. To turn this into an open-set image classification problem, the model will be trained on the 10 MNIST classes. The “unknown” classes will come from the Omniglot dataset which contains handwritten characters from 1623 classes.
    - Application 3: Real-world color image classification
        - CIFAR-100 dataset contains 32x32 color images of various real-world objects such as flowers, cars, furniture, etc. and contains examples from 100 classes. To turn this into an open-set image classification problem, 50 of the classes will be the “unknown” classes and the model will see the other 50 classes during training. 

</p>
</details>

<details><summary>Analysis and Evaluation</summary>
<p>

- An important part of this project was evaluating the performance of my models and efficacy of my approach for open-set image classification. Analysis was done as follows:
    - First, I measured accuracy when distinguishing between known and unknown classes for various threshold values used by the threshold function. This boils down to a binary classification problem: given an input, did the model correctly classify it as known/unknown? When analyzing this as a binary classification problem, I computed True positive rate, True negative rate, False positive rate, False negative rate for various thresholds. With these, I also plotted an ROC curve and DET curve to succinctly describe the aforementioned rates for various thresholds used by the threshold function. 
    - Second, it is important to measure known-class identification accuracy: for the inputs correctly labeled as one of the “known” classes, did it label it as the correct “known” class? I produced a confusion matrix here, which shows which classes were confused easily with each other. 
    - Finally, I measured top-N accuracy for known-class identification accuracy. For example, top-3 accuracy is the accuracy of the model where the true class label matches the model’s top 3 predictions. I was able to see for which N (if any), the models reach 100% top-N accuracy. 
 
</p>
</details>


## Tools

- Python3 environment
- Python libraries: 
    - ```tensorflow```, ```tensorflow_datasets```, ```keras```
    - ```numpy```
    - ```matplotlib```
    - ```sklearn```
    - ```seaborn```
    - ```scipy```

## Usage
**Run Fashion MNIST demo**

```
$ python3 fashion_mnist.py
```

**Run MNIST demo**

```
$ python3 mnist.py
```

**Run CIFAR-100 demo**

```
$ python3 cifar.py
```

**Description**

- These 3 demo scripts do the following...
    - Load data from their respective datasets
    - Pre-process the data
    - Create new train and test splits for the open-set image classification task
    - Load the respective best models
        - ```fashion_mnist.py``` loads ```saved_models/fashion_mnist.h5```
        - ```mnist.py``` loads ```saved_models/mnist_cnn.h5```
        - ```cifar.py``` loads ```saved_models/cifar_cnn.h5```
    - Perform analysis on said models
    - Output analysis graphs into their respective folders
        - ```fashion_mnist.py``` outputs to ```plots/fashion_mnist```
        - ```mnist.py``` outputs to ```plots/mnist```
        - ```cifar.py``` outputs to ```plots/cifar```

### References

- Fashion MNIST
    - https://www.tensorflow.org/datasets/catalog/fashion_mnist
- MNIST
    - https://www.tensorflow.org/datasets/catalog/mnist
- Omniglot
    - https://www.tensorflow.org/datasets/catalog/omniglot
- CIFAR-100
    - https://www.tensorflow.org/datasets/catalog/cifar100
    - https://www.tensorflow.org/api_docs/python/tf/keras/datasets/cifar100/load_data
- Model development
    - https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
    - https://keras.io/api/optimizers/
    - https://keras.io/api/layers/convolution_layers/convolution2d/
- Analysis
    - https://medium.datadriveninvestor.com/confusion-matric-tpr-fpr-fnr-tnr-precision-recall-f1-score-73efa162a25f
    - https://ccc.inaoep.mx/~villasen/bib/martin97det.pdf
    - https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc#:~:text=An%20ROC%20curve%20(receiver%20operating,False%20Positive%20Rate
    - https://towardsdatascience.com/understanding-top-n-accuracy-metrics-8aa90170b35#:~:text=What%20is%20top%20N%20accuracy,predicted%20by%20the%20classification%20model.
    - https://www.sciencedirect.com/topics/engineering/confusion-matrix#:~:text=A%20confusion%20matrix%20is%20a,performance%20of%20a%20classification%20algorithm.
# Open-Set Image Classification

## Author

Justin Lee (UNI: jjl2245)

## Project Summary

asdfasdf

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
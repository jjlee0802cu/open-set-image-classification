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
asdfasdf
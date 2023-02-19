import numpy
import tensorflow_datasets as tfds

def load_keras_dataset(dataset):
    (train_x, train_y), (test_x, test_y) = dataset.load_data()

    # Convert all to float types
    train_x = train_x.astype(float)
    train_y = train_y.astype(float)
    test_x = test_x.astype(float)
    test_y = test_y.astype(float)

    return train_x, train_y, test_x, test_y

def print_counts_from_numpy(x):
    elements, frequency = numpy.unique(x, return_counts=True)
    print(numpy.asarray((elements, frequency)).T)


def load_omniglot(split='train'):
    omniglot_x, omniglot_y = [], []
    ds = tfds.load("omniglot", split=split, as_supervised=True)
    for image, label in ds:
        omniglot_x.append(image)
        omniglot_y.append(label.numpy())
    return omniglot_x, omniglot_y
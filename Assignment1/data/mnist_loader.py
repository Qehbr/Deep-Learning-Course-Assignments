import numpy as np
from torchvision import datasets, transforms

def load_mnist():
    """
    Loads and preprocesses the MNIST dataset.

    The function downloads the MNIST training and test datasets, normalizes the pixel values,
    reshapes the data for use in a neural network, and converts the labels to one-hot encoding.

    Returns
    -------
    X_train : ndarray
        Training data of shape (784, number_of_training_examples), normalized to [0, 1]
    y_train_oh : ndarray
        One-hot encoded training labels of shape (10, number_of_training_examples)
    X_test : ndarray
        Test data of shape (784, number_of_test_examples), normalized to [0, 1]
    y_test_oh : ndarray
        One-hot encoded test labels of shape (10, number_of_test_examples)
    """
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    X_train = train_dataset.data.numpy().astype(np.float32) / 255.0
    Y_train = train_dataset.targets.numpy()
    X_test = test_dataset.data.numpy().astype(np.float32) / 255.0
    Y_test = test_dataset.targets.numpy()


    X_train = X_train.reshape(X_train.shape[0], -1).T
    X_test = X_test.reshape(X_test.shape[0], -1).T

    y_train_oh = np.eye(10)[Y_train].T
    y_test_oh = np.eye(10)[Y_test].T

    return X_train, y_train_oh, X_test, y_test_oh


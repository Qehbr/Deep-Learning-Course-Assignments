import numpy as np
from torchvision import datasets, transforms

def load_mnist():
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


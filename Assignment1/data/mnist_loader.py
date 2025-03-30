import numpy as np
from torchvision import datasets, transforms

def load_mnist(flatten=True, validation_split=0.2):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    x_train = train_dataset.data.numpy().astype(np.float32) / 255.0
    y_train = train_dataset.targets.numpy()
    x_test = test_dataset.data.numpy().astype(np.float32) / 255.0
    y_test = test_dataset.targets.numpy()

    if flatten:
        x_train = x_train.reshape(x_train.shape[0], -1).T
        x_test = x_test.reshape(x_test.shape[0], -1).T

    y_train_oh = np.eye(10)[y_train].T
    y_test_oh = np.eye(10)[y_test].T

    m = x_train.shape[1]
    val_size = int(m * validation_split)
    indices = np.random.permutation(m)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    x_val = x_train[:, val_indices]
    y_val_oh = y_train_oh[:, val_indices]
    x_train = x_train[:, train_indices]
    y_train_oh = y_train_oh[:, train_indices]

    return x_train, y_train_oh, x_val, y_val_oh, x_test, y_test_oh

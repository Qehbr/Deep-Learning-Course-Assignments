import numpy as np


def initialize_parameters(layer_dims):
    """
    Initializes weights and biases for each layer in a neural network.

    Parameters
    ----------
    layer_dims : list of int
        An array of the dimensions of each layer in the network (layer 0 is the size of the flattened input, layer L is the output softmax)

    Returns
    -------
    dict
        A dictionary containing the initialized W and b parameters of each layer (W1…WL, b1…bL).
    """
    parameters = {}

    for l in range(1, len(layer_dims)):
        parameters[f"W{l}"] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.1
        parameters[f"b{l}"] = np.zeros((layer_dims[l], 1))

    return parameters

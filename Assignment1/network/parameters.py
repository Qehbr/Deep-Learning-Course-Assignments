import numpy as np

def initialize_parameters(layer_dims):
    """
    Initializes weights and biases for each layer in the network.

    Args:
        layer_dims (list): Dimensions of each layer in the network.

    Returns:
        dict: Dictionary containing W1...WL and b1...bL.
    """

    parameters = {}

    for l in range(1, len(layer_dims)):
        parameters[f"W{l}"] = np.random.randn(layer_dims[l], layer_dims[l - 1])
        parameters[f"b{l}"] = np.zeros((layer_dims[l], 1))

    return parameters

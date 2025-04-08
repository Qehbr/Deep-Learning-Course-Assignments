import numpy as np


def relu(Z):
    """
    Applies the ReLU activation function.

    Parameters
    ----------
    Z : ndarray
        The linear component of the activation function

    Returns
    -------
    A : ndarray
        The activations of the layer
    activation_cache : dict
        Returns Z, which will be useful for the backpropagation
    """
    A = np.maximum(0, Z)
    return A, {'Z': Z}


def softmax(Z):
    """
    Applies the softmax activation function.

    Parameters
    ----------
    Z : ndarray
        The linear component of the activation function

    Returns
    -------
    A : ndarray
        The activations of the layer
    activation_cache : dict
        Returns Z, which will be useful for the backpropagation
    """
    Z_shifted = Z - np.max(Z, axis=0, keepdims=True)
    exp_Z = np.exp(Z_shifted)
    A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
    return A, {'Z': Z}


def relu_backward(dA, activation_cache):
    """
    Implements backward propagation for a ReLU unit.

    Parameters
    ----------
    dA : ndarray
        The post-activation gradient
    activation_cache : dict
        Contains Z (stored during the forward propagation)

    Returns
    -------
    dZ : ndarray
        Gradient of the cost with respect to Z
    """
    Z = activation_cache['Z']
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


def softmax_backward(dA, activation_cache):
    """
    Implements backward propagation for a softmax unit.

    Parameters
    ----------
    dA : ndarray
        The post-activation gradient (p_i - y_i)
    activation_cache : dict
        Contains Z (stored during the forward propagation)

    Returns
    -------
    dZ : ndarray
        Gradient of the cost with respect to Z
    """
    return dA

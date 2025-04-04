import numpy as np

def relu(Z):
    """
    Applies ReLU activation.
    """
    A = np.maximum(0, Z)
    return A, {'Z': Z}

def softmax(Z):
    """
    Applies softmax activation.
    """
    Z_shifted = Z - np.max(Z, axis=0, keepdims=True)
    exp_Z = np.exp(Z_shifted)
    A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
    return A, {'Z': Z}

def relu_backward(dA, activation_cache):
    """
    Backward propagation for a ReLU unit.
    """
    Z = activation_cache['Z']
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


def softmax_backward(dA, activation_cache):
    """
    Backward propagation for a softmax unit with cross-entropy loss.
    """
    return dA

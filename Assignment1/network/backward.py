import numpy as np
from .activations import relu_backward, softmax_backward


def linear_backward(dZ, cache):
    """
    Linear part of backward propagation for a single layer.
    """
    A_prev, W, b = cache['A'], cache['W'], cache['b']
    m = A_prev.shape[1]
    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    """
    Backward propagation for LINEAR->ACTIVATION layer.
    """
    linear_cache = {'A': cache['A'], 'W': cache['W'], 'b': cache['b']}
    activation_cache = {'Z': cache['Z']}
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    elif activation == "softmax":
        dZ = softmax_backward(dA, activation_cache)
    else:
        raise ValueError(f"Unsupported activation function: '{activation}'. Expected 'relu' or 'softmax'.")
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db


def l_model_backward(AL, Y, caches):
    """
    Backward propagation for the entire network.
    """
    grads = {}
    L = len(caches)

    dAL = AL - Y
    current_cache = caches[L - 1]
    grads[f"dA{L}"], grads[f"dW{L}"], grads[f"db{L}"] = linear_activation_backward(dAL, current_cache, "softmax")

    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev, dW, db = linear_activation_backward(grads[f"dA{l + 2}"], current_cache, "relu")
        grads[f"dA{l + 1}"] = dA_prev
        grads[f"dW{l + 1}"] = dW
        grads[f"db{l + 1}"] = db

    return grads

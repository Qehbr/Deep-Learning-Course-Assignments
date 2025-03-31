# File: network/backward.py
import numpy as np
from .activations import relu_backward, softmax_backward
# Make sure to import the new batchnorm function
from .batchnorm import batchnorm_backward # <-- ADD THIS IMPORT


def linear_backward(dZ, cache, lambd=0.0):
    """
    Linear part of backward propagation for a single layer.
    (No changes needed here itself, but ensure 'cache' contains A, W, b)
    """
    A_prev, W, b = cache['A'], cache['W'], cache['b']
    m = A_prev.shape[1]
    if lambd != 0:
        # Note: Regularization is applied to W before BN affects Z
        dW = (np.dot(dZ, A_prev.T) / m) + (lambd / m) * W
    else:
        dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation, lambd=0.0):
    """
    Backward propagation for the LINEAR->[BATCHNORM]->ACTIVATION layer.
    """
    # Unpack the caches stored during the forward pass
    linear_cache = cache['linear_cache']
    bn_cache = cache['bn_cache']
    activation_cache = cache['activation_cache']

    # 1. Backward pass through Activation
    if activation == "relu":
        # dA is gradient w.r.t activation output A
        # dZ_output is gradient w.r.t activation input Z (which is Z_norm if BN was used)
        dZ_output = relu_backward(dA, activation_cache)
    elif activation == "softmax":
        # dA is the initial gradient (AL-Y) calculated in l_model_backward
        # This dA represents dZ for the final layer when combined with cross-entropy
        dZ_output = softmax_backward(dA, activation_cache) # Assumes softmax_backward returns dA
    else:
        raise ValueError(f"Unsupported activation function: '{activation}'.")

    # 2. Backward pass through Batch Normalization (Optional)
    if bn_cache is not None:
        # dZ_output is gradient w.r.t BN output (Z_norm)
        # dZ is gradient w.r.t BN input (the original Z from linear step)
        dZ = batchnorm_backward(dZ_output, bn_cache) # <-- Use BN backward pass
    else:
        # If no BN, the gradient w.r.t activation input is the gradient w.r.t linear output
        dZ = dZ_output

    # 3. Backward pass through Linear Layer
    # dZ is gradient w.r.t linear output
    # dA_prev, dW, db are gradients w.r.t linear input, weights, bias
    dA_prev, dW, db = linear_backward(dZ, linear_cache, lambd=lambd)

    return dA_prev, dW, db


def l_model_backward(AL, Y, caches, lambd=0.0):
    """
    Backward propagation for the entire network.
    (Minor change to handle cache structure if needed, and ensure correct dA is passed)
    """
    grads = {}
    L = len(caches) # Number of layers

    # Initial gradient (dA for the last layer activation)
    # For softmax + cross-entropy, dZ_output = AL - Y.
    # We pass this as dA to linear_activation_backward for the last layer.
    dAL = AL - Y

    # Last Layer (Softmax, typically no BN)
    current_cache = caches[L - 1]
    # linear_activation_backward expects dA (gradient w.r.t activation output)
    # For softmax/cross-entropy, we start with dZ = AL-Y. We pass this as dA.
    # softmax_backward (as currently written) just returns this, resulting in dZ_output = AL-Y.
    # Since BN is usually off for the last layer, dZ = dZ_output.
    # Then linear_backward calculates dA_prev, dW, db using this dZ.
    grads[f"dA{L}"], grads[f"dW{L}"], grads[f"db{L}"] = linear_activation_backward(
        dAL, current_cache, "softmax", lambd
    ) # dA{L} here is actually dA_prev for layer L-1

    # Loop through hidden layers (ReLU, potentially with BN)
    for l in reversed(range(L - 1)): # From L-2 down to 0
        current_cache = caches[l]
        # The gradient needed is dA for layer l+1's activation output.
        # This is stored as grads[f"dA{l+2}"] from the previous iteration (dA_prev of layer l+1)
        dA_input = grads[f"dA{l + 2}"] # dA{l+2} is dA_prev from layer l+1, which is dA for layer l+1 activation

        dA_prev, dW, db = linear_activation_backward(
            dA_input, current_cache, "relu", lambd
        )

        # Store gradients
        grads[f"dA{l + 1}"] = dA_prev # dA_prev for layer l
        grads[f"dW{l + 1}"] = dW      # dW for layer l+1
        grads[f"db{l + 1}"] = db      # db for layer l+1

    return grads

# No changes needed in activations.py, loss.py, optimization.py, parameters.py,
# predict.py, utils.py, mnist_loader.py, or the main run script,
# assuming they correctly call the modified forward/backward functions.
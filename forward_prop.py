import numpy as np

def initialize_parameters(layer_dims):
    """
    Initialize parameters for a neural network

    Arguments:
    layer_dims -- python list containing the dimensions of each layer in the network

    Returns:
    params_dict -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
    """

    params_dict = {}
    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):
        params_dict['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        params_dict['b' + str(l)] = np.zeros((layer_dims[l], 1))
        assert (params_dict['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert (params_dict['b' + str(l)].shape == (layer_dims[l], 1))

    return params_dict

def linear_forward(A, W, b):
    
    """
    Linear Forward propagation

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples in the current layer)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of current layer, 1)

    Returns:
    Z -- the input of the activation function
    """
    linear_cache = {'A': A, 'W': W, 'b': b}

    Z = W @ A + b

    return Z, linear_cache

def softmax(Z: np.ndarray):

    z_sum = np.exp(Z).sum()

    A = np.exp(Z) / z_sum

    activation_cache = Z
    
    return A, activation_cache

def relu(Z):

    A = np.maximum(0, Z)

    activation_cache = Z
    
    return A, activation_cache

def linear_activation_forward(A_prev, W, B, activation):
    
    if activation == "softmax":
        act_func = softmax
    elif activation == "relu":
        act_func = relu
    else:
        raise ValueError("activation needs to be softmax or relu")
    
    Z, cache = linear_forward(A_prev, W, B)

    A, _ = act_func(Z)

    cache['Z'] = Z

    return A, cache

def l_model_forward(X, parameters, use_batchnorm):

    l = 1
    A = X
    caches = []

    while ('W' + str(l) in parameters):
        w = parameters['W' + str(l)]
        b = parameters['b' + str(l)]

        activation = "relu"
        if 'W' + str(l + 1) not in parameters:
            activation = "softmax"
        A, cache = linear_activation_forward(A, w, b, activation)
        if use_batchnorm:
            A = apply_batchnorm(A)
        caches.append(cache)

        l = l + 1

    AL = A

    return AL, caches

def compute_cost(AL, Y): # TODO: fix

    AL_log = np.log(AL)
    m = AL.shape[1]
    return -(1/m) * (Y @ AL_log).sum()


def apply_batchnorm(A): # TODO: implement
    return
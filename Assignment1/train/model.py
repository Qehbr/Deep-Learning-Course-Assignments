import numpy as np
from network.parameters import initialize_parameters
from network.forward import l_model_forward
from network.backward import l_model_backward
from network.loss import compute_cost
from network.optimization import update_parameters
from .utils import create_minibatches


def l_layer_model(X, Y, layers_dims, learning_rate, num_iterations,
                  batch_size, use_batchnorm, lambd=0.0, random_seed=1):
    np.random.seed(random_seed)
    costs = []
    parameters = initialize_parameters(layers_dims)

    for i in range(num_iterations):
        minibatches = create_minibatches(X, Y, batch_size)
        epoch_cost = 0

        for minibatch_X, minibatch_Y in minibatches:
            AL, caches = l_model_forward(minibatch_X, parameters, use_batchnorm)
            cost = compute_cost(AL, minibatch_Y, parameters, lambd)
            grads = l_model_backward(AL, minibatch_Y, caches)
            parameters = update_parameters(parameters, grads, learning_rate)
            epoch_cost += cost / len(minibatches)

        costs.append((i, epoch_cost))
        print(f"Cost after iteration {i}: {epoch_cost:.4f}")

    return parameters, costs

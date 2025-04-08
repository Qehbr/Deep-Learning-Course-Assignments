import numpy as np
import matplotlib.pyplot as plt


def create_minibatches(X, Y, batch_size):
    """
    Creates randomized minibatches from the training data.

    Parameters
    ----------
    X : ndarray
        Input data of shape (input_size, number_of_examples)
    Y : ndarray
        One-hot encoded labels of shape (num_classes, number_of_examples)
    batch_size : int
        Number of examples in each minibatch

    Returns
    -------
    minibatches : list of tuple
        A list where each element is a tuple (X_batch, Y_batch) representing a minibatch
    """
    m = X.shape[1]
    permutation = np.random.permutation(m)
    X_shuffled = X[:, permutation]
    Y_shuffled = Y[:, permutation]

    minibatches = []
    num_complete_minibatches = m // batch_size
    for k in range(num_complete_minibatches):
        X_batch = X_shuffled[:, k * batch_size:(k + 1) * batch_size]
        Y_batch = Y_shuffled[:, k * batch_size:(k + 1) * batch_size]
        minibatches.append((X_batch, Y_batch))

    if m % batch_size != 0:
        X_batch = X_shuffled[:, num_complete_minibatches * batch_size:]
        Y_batch = Y_shuffled[:, num_complete_minibatches * batch_size:]
        minibatches.append((X_batch, Y_batch))

    return minibatches


def plot_costs(costs):
    """
   Plots training and validation cost over training iterations.

   Parameters
   ----------
   costs : list of tuple
       List of tuples in the form (iteration, avg_train_cost, val_cost)
   """

    steps, train_costs, val_costs = zip(*costs)
    plt.plot(steps, train_costs, label="Train cost")
    plt.plot(steps, val_costs, label="Validation cost")
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Training and Validation Cost over Iterations")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_weights_distribution(parameters):
    """
    Plots histograms for the weight distributions of each layer.

    Parameters
    ----------
    parameters : dict
        A dictionary containing the trained weights (W1...WL) and biases (b1...bL) of the model
    """
    L = len(parameters) // 2  # number of layers
    for l in range(1, L + 1):
        W = parameters[f"W{l}"]

        plt.figure()
        plt.hist(W.flatten(), bins=50)
        plt.title(f"Distribution of W{l}")
        plt.xlabel("Weight value")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()

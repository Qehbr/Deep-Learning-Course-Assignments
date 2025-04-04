import numpy as np
import matplotlib.pyplot as plt


def create_minibatches(X, Y, batch_size):
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
    """
    L = len(parameters) // 2  # number of layers
    for l in range(1, L + 1):
        W = parameters[f"W{l}"]

        plt.figure()  # ensure each histogram is on its own figure
        plt.hist(W.flatten(), bins=50)  # e.g., 50 bins
        plt.title(f"Distribution of W{l}")
        plt.xlabel("Weight value")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()

import numpy as np

from network.forward import l_model_forward


def predict(X, Y, parameters, use_batchnorm):
    """
    Calculates the accuracy of the trained neural network on the provided data.

    Parameters
    ----------
    X : ndarray
        The input data, a numpy array of shape (height * width, number_of_examples)
    Y : ndarray
        The true labels of the data, a one-hot encoded array of shape (num_of_classes, number_of_examples)
    parameters : dict
        A dictionary containing the DNN architecture’s parameters
    use_batchnorm : bool
        If True, applies batch normalization during forward propagation

    Returns
    -------
    accuracy : float
        The accuracy of the model on the input data, computed as the percentage of
        samples for which the correct label receives the highest confidence score
    """
    AL, _ = l_model_forward(X, parameters, use_batchnorm=use_batchnorm)
    predictions = np.argmax(AL, axis=0)
    labels = np.argmax(Y, axis=0)
    accuracy = np.mean(predictions == labels)
    return accuracy

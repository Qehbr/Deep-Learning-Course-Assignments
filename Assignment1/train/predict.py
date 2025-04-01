import numpy as np

from network.forward import l_model_forward


def predict(X, Y, parameters, use_batchnorm):
    AL, _ = l_model_forward(X, parameters, use_batchnorm=use_batchnorm)
    predictions = np.argmax(AL, axis=0)
    labels = np.argmax(Y, axis=0)
    accuracy = np.mean(predictions == labels)
    return accuracy
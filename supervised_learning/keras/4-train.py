#!/usr/bin/env python3
"""This module contains the function train_model"""


def train_model(
        network,
        data,
        labels,
        batch_size,
        epochs,
        verbose=True,
        shuffle=False):
    """trains a model using mini-batch gradient descent

    Params:
        network: the model to train
        data: a one-hot numpy.ndarray of shape (m, classes)
              containing the labels of data
        batch_size: the size of the batch used for mini-batch gradient descent
        epochs: the number of passes through data for mini-batch gradient
                descent
        verbose: a boolean that determines if output should be printed
                 during training
        shuffle: a boolean that determines whether to shuffle the batches
                 every epoch. Normally, it is a good idea to shuffle, but
                 for reproducibility, we have chosen to set the default to
                 False

    Returns: the History object generated after training the model
    """
    history = network.fit(data,
                          labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=verbose,
                          shuffle=shuffle)
    return history

#!/usr/bin/env python3
"""This module contains the function train_model"""
import tensorflow.keras as K


def train_model(
        network,
        data,
        labels,
        batch_size,
        epochs,
        validation_data=None,
        early_stopping=False,
        patience=0,
        learning_rate_decay=False,
        alpha=0.1,
        decay_rate=1,
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
        learning_rate_decay: a boolean that indicates whether learning rate
                             decay should be used
        alpha: the initial learning rate
        decay_rate: the decay rate

    Returns: the History object generated after training the model
    """
    callbacks = []
    if early_stopping and validation_data:
        es = K.callbacks.EarlyStopping(
            monitor='val_loss', mode='min', patience=patience)
        callbacks.append(es)

    if learning_rate_decay and validation_data:
        learning_rate_fn = K.optimizers.schedules.InverseTimeDecay(
            initial_learning_rate=alpha,
            decay_steps=1,
            decay_rate=decay_rate,
            staircase=True)
        lrd = K.callbacks.LearningRateScheduler(learning_rate_fn, verbose=True)
        callbacks.append(es)

    history = network.fit(data,
                          labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          callbacks=callbacks,
                          verbose=verbose,
                          shuffle=shuffle,
                          validation_data=validation_data)
    return history

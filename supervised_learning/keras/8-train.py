#!/usr/bin/env python3
"""This module contains the function train_model"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False, patience=0, learning_rate_decay=False, alpha=0.1, decay_rate=1, save_best=False, filepath=None, verbose=True, shuffle=False):
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
        save_best : a boolean indicating whether to save the model after each
                    epoch if it is the best a model is considered the best if 
                    its validation loss is the lowest that the model has 
                    obtained
        filepath: the file path where the model should be saved

    Returns: the History object generated after training the model
    """
    callbacks = []
    if early_stopping and validation_data:
        es = K.callbacks.EarlyStopping(
            monitor='val_loss', mode='min', patience=patience)
        callbacks.append(es)

    def learning_rate_fn(epoch):
        """The function that sets the learning rate for each epoch"""
        return alpha / (1 + decay_rate * epoch)

    if learning_rate_decay and validation_data:
        lrd = K.callbacks.LearningRateScheduler(learning_rate_fn, verbose=1)
        callbacks.append(lrd)

    if save_best:
        best = K.callbacks.ModelCheckpoint(filepath=filepath,
                                           save_best_only=True,
                                           mode='min')
        callbacks.append(best)

    history = network.fit(data,
                          labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          callbacks=callbacks,
                          verbose=verbose,
                          shuffle=shuffle,
                          validation_data=validation_data)
    return history

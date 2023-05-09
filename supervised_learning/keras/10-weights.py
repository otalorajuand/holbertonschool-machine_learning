#!/usr/bin/env python3
"""This module contains the functions save_weights and load_weights"""


def save_weights(network, filename, save_format='h5'):
    """saves a model’s weights

    Params:
        network: the model whose weights should be saved
        filename: the path of the file that the weights should be saved to
        save_format: the format in which the weights should be saved

    Returns: None
    """
    network.save_weights(filename, save_format=save_format)


def load_weights(network, filename):
    """loads a model’s weights

    Params:
        network: the model to which the weights should be loaded
        filename: the path of the file that the weights should be loaded from

    Returns: None
    """
    network.load_weights(filename)

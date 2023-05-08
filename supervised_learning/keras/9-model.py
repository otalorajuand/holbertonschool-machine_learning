#!/usr/bin/env python3
"""This module contains the function save_model and load_model"""
import tensorflow.keras as K


def save_model(network, filename):
    """saves an entire model

    Params:
        network: the model to save
        filename: the path of the file that the model should be saved to

    Returns: None
    """
    network.save(filename)


def load_model(filename):
    """loads an entire model

    Params:
        filename: the path of the file that the model should be loaded from

    Returns: the loaded model
    """
    model = K.models.load_model(filename)
    return model

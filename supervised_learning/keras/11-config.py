#!/usr/bin/env python3
"""This module contains save_config and load_config"""
import tensorflow.keras as K


def save_config(network, filename):
    """saves a model’s configuration in JSON format

    Params:
        network: the model whose configuration should be saved
        filename: the path of the file that the configuration should be
                  saved to

    Returns: None
    """
    json_config = network.to_json()

    with open(filename, "w") as json_file:
        json_file.write(json_config)


def load_config(filename):
    """loads a model with a specific configuration

    Params:
        filename: the path of the file containing the model's
                  configuration in JSON format

    Returns: the loaded model
    """
    with open(filename, 'r') as json_file:
        loaded_model_json = json_file.read()
    model = K.models.model_from_json(loaded_model_json)

    return model

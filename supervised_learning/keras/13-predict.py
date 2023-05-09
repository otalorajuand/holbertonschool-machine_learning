#!/usr/bin/env python3
"""This module contains the function predict"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """makes a prediction using a neural network

    Params:
        network: the network model to make the prediction with
        data: the input data to make the prediction with
        verbose: a boolean that determines if output should be
                 printed during the prediction process

    Returns: the prediction for the data
    """
    prediction = network.predict(data, verbose=verbose)
    return prediction

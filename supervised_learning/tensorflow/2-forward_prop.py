#!/usr/bin/env python3
"""This module contains the function forward_prop"""

create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """creates the forward propagation graph for the neural network"""
    lay = create_layer(x, layer_sizes[0], activations[0])
    for i in range(len(1, layer_sizes)):
        lay = create_layer(lay, layer_sizes[i], activations[i])

    return lay

#!/usr/bin/env python3
"""This module contains the function learning_rate_decay"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """updates the learning rate using inverse time decay in numpy"""
    res = (1/(1 + decay_rate * int(global_step / decay_step)))*alpha
    return res

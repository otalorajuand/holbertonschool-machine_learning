#!/usr/bin/env python3
"""This module contains the function from_numpy"""
import pandas as pd
import string


def from_numpy(array):
    """creates a pd.DataFrame from a np.ndarray

    Args:
    - array is the np.ndarray from which you should create the pd.DataFrame

    Returns: the newly created pd.DataFrame
    """
    cols = array.shape[1]
    d = list(string.ascii_uppercase)[:cols]
    df = pd.DataFrame(array, columns=d)

    return df

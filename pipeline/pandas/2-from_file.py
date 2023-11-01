#!/usr/bin/env python3
"""This module contains the function from_file"""
import pandas as pd


def from_file(filename, delimiter):
    """Loads data from a file as a pd.DataFrame

    Args:
    - filename is the file to load from
    - delimiter is the column separator

    Returns: the loaded pd.DataFrame
    """

    df = pd.read_csv(filename, sep=delimiter)

    return df

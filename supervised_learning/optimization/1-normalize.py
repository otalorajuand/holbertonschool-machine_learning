#!/usr/bin/env python3
"""This module contains the function normalize"""


def normalize(X, m, s):
    """normalizes (standardizes) a matrix"""
    return (X - m)/s

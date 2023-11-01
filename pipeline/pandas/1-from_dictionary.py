#!/usr/bin/env python3
"""This file contains the object df"""
import pandas as pd


df = pd.DataFrame(
    {
        "First": [0.0, 0.5, 1.0, 1.5],
        "Second": ["one", "two", "three", "four"]
    }
)

df.index = ['A', 'B', 'C', 'D']

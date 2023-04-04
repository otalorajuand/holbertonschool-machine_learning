#!/usr/bin/env python3
"""This module contains the class Binomial"""


class Binomial:
    """This class represents a binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):
        if data is None:
            if n < 0:
                raise ValueError("n must be a positive value")
            else:
                self.n = int(n)
            if p < 0 or p > 1:
                raise ValueError("p must be greater than 0 and less than 1")
            else:
                self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            x_bar = sum(data) / len(data)
            s2 = sum((x - x_bar)**2 for x in data) / (len(data) - 1)
            self.n = round((x_bar ** 2) / (x_bar - s2))
            self.p = x_bar / self.n

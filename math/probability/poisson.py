#!/usr/bin/env python3
"""This module contains the class Poisson"""


class Poisson:
    """This class represents a poisson distribution"""
    def __init__(self, data=None, lambtha=1.):
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            else:
                self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data)/len(data))

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of successes"""
        e = 2.7182818285
        if type(k) is not int:
            k = int(k)

        if k < 0:
            return 0

        fact = 1
        for i in range(1, k+1):
            fact = fact * i

        return ((e**(-self.lambtha))*(self.lambtha**k))/fact

#!/usr/bin/env python3
import numpy as np
Binomial = __import__('binomial').Binomial

np.random.seed(6)
p = np.random.uniform(0.01, 0.99)
n = np.random.randint(1, 100)
s = np.random.randint(100, 1000)
data = np.random.binomial(n, p, s).tolist()
b = Binomial()
print(b.n, b.p)
b = Binomial(data)
print(b.n, np.around(b.p, 10))
b = Binomial(n=n, p=p)
print(b.n, np.around(b.p, 10))

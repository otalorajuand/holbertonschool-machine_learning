#!/usr/bin/env python3

import numpy as np
Neuron = __import__('5-neuron').Neuron

np.random.seed(6)
nx, m = np.random.randint(100, 1000, 2).tolist()
nn = Neuron(nx)
X = np.random.randn(nx, m)
Y = np.random.randint(0, 2, (1, m))
A = np.random.uniform(size=(1, m))
print(nn.W)
nn.gradient_descent(X, Y, A)
print(nn.W)
nn.gradient_descent(X, Y, A, alpha=0.5)
print(nn.W)
try:
    nn.W = 10
    print('Fail: private attribute W overwritten as a public attribute')
except:
    pass
